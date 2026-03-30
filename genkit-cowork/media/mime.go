// Copyright 2026 Kevin Lopes
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package media

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/gif"
	"image/jpeg"
	"image/png"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"golang.org/x/image/draw"
	"golang.org/x/image/webp"
)

var imageMimeTypes = []string{
	"image/jpeg",
	"image/png",
	"image/gif",
	"image/webp",
}

const (
	// MaxWidth is the maximum output image width in pixels.
	MaxWidth = 2000
	// MaxHeight is the maximum output image height in pixels.
	MaxHeight = 2000
	// MaxBytes is the maximum output image size in bytes.
	MaxBytes    = 5 * 1024 * 1024 // 5MB
	jpegQuality = 80
)

func isValidImageMimeType(mimeType string) bool {
	if mimeType == "" {
		return false
	}
	return slices.Contains(imageMimeTypes, mimeType)
}

// DetectImageMimeType returns the MIME type of a file if it's a valid image type, or "" otherwise.
func DetectImageMimeType(filePath string) string {
	contentType := DetectMimeType(filePath)
	if isValidImageMimeType(contentType) {
		return contentType
	}
	return ""
}

// DetectMimeType reads the first 512 bytes of a file and returns its MIME type.
func DetectMimeType(filePath string) string {
	file, err := os.Open(filePath)
	if err != nil {
		return ""
	}
	defer file.Close()

	buffer := make([]byte, 512)
	n, err := file.Read(buffer)
	if err != nil || n == 0 {
		return ""
	}

	contentType := http.DetectContentType(buffer[:n])
	if strings.HasPrefix(contentType, "text/plain") || contentType == "application/octet-stream" {
		ext := strings.ToLower(filepath.Ext(filePath))
		switch ext {
		case ".md", ".markdown":
			contentType = "text/markdown"
		case ".js", ".json":
			contentType = "application/json"
		case ".html", ".htm":
			contentType = "text/html"
		case ".csv":
			contentType = "text/csv"
		}
	}
	return contentType
}

// ResizeResult holds the output of an AutoResizeImage call.
type ResizeResult struct {
	// Data is the raw image bytes (original or re-encoded).
	Data []byte
	// Base64 is the base64-encoded image data.
	Base64 string
	// MimeType is the MIME type of the output image.
	// May differ from the input when re-encoding is needed.
	MimeType string
	// OriginalWidth is the width of the input image.
	OriginalWidth int
	// OriginalHeight is the height of the input image.
	OriginalHeight int
	// FinalWidth is the width of the output image.
	FinalWidth int
	// FinalHeight is the height of the output image.
	FinalHeight int
	// WasResized indicates whether the image was scaled down.
	WasResized bool
}

// AutoResizeImage decodes an image from raw bytes, checks whether it exceeds
// the maximum dimensions (2000x2000) or byte size (5MB), and scales it down
// if needed. The output is always JPEG-encoded when resizing occurs.
//
// Supported input formats: JPEG, PNG, GIF, WebP.
func AutoResizeImage(imageData []byte, mimeType string) (*ResizeResult, error) {
	reader := bytes.NewReader(imageData)

	var src image.Image
	var err error

	switch mimeType {
	case "image/jpeg":
		src, err = jpeg.Decode(reader)
	case "image/png":
		src, err = png.Decode(reader)
	case "image/gif":
		src, err = gif.Decode(reader)
	case "image/webp":
		src, err = webp.Decode(reader)
	default:
		return nil, fmt.Errorf("unsupported image format: %s", mimeType)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to decode %s image: %w", mimeType, err)
	}

	bounds := src.Bounds()
	origWidth := bounds.Dx()
	origHeight := bounds.Dy()

	// Check if resizing is needed.
	needsResize := origWidth > MaxWidth || origHeight > MaxHeight || len(imageData) > MaxBytes

	if !needsResize {
		encoded := base64.StdEncoding.EncodeToString(imageData)
		return &ResizeResult{
			Data:           imageData,
			Base64:         encoded,
			MimeType:       mimeType,
			OriginalWidth:  origWidth,
			OriginalHeight: origHeight,
			FinalWidth:     origWidth,
			FinalHeight:    origHeight,
			WasResized:     false,
		}, nil
	}

	// Calculate new dimensions preserving aspect ratio.
	newWidth, newHeight := fitDimensions(origWidth, origHeight, MaxWidth, MaxHeight)

	// Scale the image using CatmullRom interpolation for quality.
	dst := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))
	draw.CatmullRom.Scale(dst, dst.Bounds(), src, bounds, draw.Over, nil)

	// Encode as JPEG.
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, dst, &jpeg.Options{Quality: jpegQuality}); err != nil {
		return nil, fmt.Errorf("failed to encode resized image: %w", err)
	}

	resizedData := buf.Bytes()
	encoded := base64.StdEncoding.EncodeToString(resizedData)

	return &ResizeResult{
		Data:           resizedData,
		Base64:         encoded,
		MimeType:       "image/jpeg",
		OriginalWidth:  origWidth,
		OriginalHeight: origHeight,
		FinalWidth:     newWidth,
		FinalHeight:    newHeight,
		WasResized:     true,
	}, nil
}

// FormatDimensionNote returns a human-readable note about image dimensions
// and any resizing that occurred. Returns "" if no resizing happened.
func FormatDimensionNote(result *ResizeResult) string {
	if !result.WasResized {
		return ""
	}
	return fmt.Sprintf("Resized from %dx%d to %dx%d",
		result.OriginalWidth, result.OriginalHeight,
		result.FinalWidth, result.FinalHeight)
}

// fitDimensions calculates new dimensions that fit within maxW x maxH
// while preserving the original aspect ratio.
func fitDimensions(width, height, maxW, maxH int) (int, int) {
	if width <= maxW && height <= maxH {
		return width, height
	}

	ratioW := float64(maxW) / float64(width)
	ratioH := float64(maxH) / float64(height)

	ratio := ratioW
	if ratioH < ratioW {
		ratio = ratioH
	}

	newWidth := int(float64(width) * ratio)
	newHeight := int(float64(height) * ratio)

	// Ensure at least 1x1.
	if newWidth < 1 {
		newWidth = 1
	}
	if newHeight < 1 {
		newHeight = 1
	}

	return newWidth, newHeight
}

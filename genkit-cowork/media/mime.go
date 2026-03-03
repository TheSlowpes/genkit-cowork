package media

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/jpeg"
	"net/http"
	"os"
	"slices"
)

var imageMimeTypes = []string{
	"image/jpeg",
	"image/png",
	"image/gif",
	"image/webp",
}

const (
	maxWidth    = 2000
	maxHeight   = 2000
	maxBytes    = 5 * 1024 * 1024 // 5MB
	jpegQuality = 80
)

func isValidImageMimeType(mimeType string) bool {
	if mimeType == "" {
		return false
	}
	return slices.Contains(imageMimeTypes, mimeType)
}

func DetectImageMimeType(filePath string) string {
	file, err := os.Open(filePath)
	if err != nil {
		return ""
	}
	defer file.Close()

	buffer := make([]byte, 512)
	_, err = file.Read(buffer)
	if err != nil {
		return ""
	}

	contentType := http.DetectContentType(buffer)
	if isValidImageMimeType(contentType) {
		return contentType
	}
	return ""
}

func AutoResizeImage(imageData []byte, mimeType string) ([]byte, string, error) {
	reader := bytes.NewReader(imageData)
	var src image.Image
	switch mimeType {
	case "image/jpeg":
		src, err := jpeg.Decode(reader)
		if err != nil {
			return nil, "", fmt.Errorf("failed to decode JPEG image: %w", err)
		}
	}
	currentWidth := src.Bounds().Dx()
	currentHeight := src.Bounds().Dy()

	if currentWidth <= maxWidth && currentHeight <= maxHeight && len(imageData) <= maxBytes {
		encodedBuffer := base64.StdEncoding.EncodeToString(imageData)
		return imageData, encodedBuffer, nil
	}
}

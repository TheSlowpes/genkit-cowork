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

// Package groundingfixture supplies paired test images for the Yzma box example.
package groundingfixture

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
)

const (
	frameEdge  = 320
	scale      = 1000
	minOverlap = 0.5
)

// Instruction is identical for both images; it does not give away the position.
const Instruction = "Find the solid red square. Give the tight bounding box around it. Ignore the blue square."

// Target specifies one position of the same red square.
type Target struct {
	Name string
	rect image.Rectangle
}

// Targets returns the left and right fixtures, each with a blue distractor.
func Targets() [2]Target {
	return [2]Target{
		{"left", image.Rect(32, 64, 128, 160)},
		{"right", image.Rect(192, 64, 288, 160)},
	}
}

// PNG returns a self-contained PNG data URL.
func (t Target) PNG() (string, error) {
	img := image.NewRGBA(image.Rect(0, 0, frameEdge, frameEdge))
	draw.Draw(img, img.Bounds(), image.White, image.Point{}, draw.Src)
	red := image.NewUniform(color.RGBA{R: 220, A: 255})
	blue := image.NewUniform(color.RGBA{B: 220, A: 255})
	distractor := image.Rect(frameEdge-t.rect.Max.X, 192, frameEdge-t.rect.Min.X, 288)
	draw.Draw(img, t.rect, red, image.Point{}, draw.Src)
	draw.Draw(img, distractor, blue, image.Point{}, draw.Src)
	var data bytes.Buffer
	if err := png.Encode(&data, img); err != nil {
		return "", err
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(data.Bytes()), nil
}

// Check requires 0.5 intersection-over-union in normalized image coordinates.
// A whole-frame prediction scores 0.09; the distractor scores zero.
func (t Target) Check(left, top, right, bottom int) error {
	want := image.Rectangle{Min: t.rect.Min.Mul(scale).Div(frameEdge), Max: t.rect.Max.Mul(scale).Div(frameEdge)}
	got := image.Rectangle{Min: image.Pt(left, top), Max: image.Pt(right, bottom)}
	if got.Empty() || !got.In(image.Rect(0, 0, scale, scale)) {
		return fmt.Errorf("%s red square: box %v is not valid image geometry", t.Name, got)
	}
	area := func(rect image.Rectangle) int { return rect.Dx() * rect.Dy() }
	intersection := area(want.Intersect(got))
	union := area(want) + area(got) - intersection
	if float64(intersection)/float64(union) < minOverlap {
		return fmt.Errorf("%s red square: box %v must overlap target %v with IoU >= %.1f", t.Name, got, want, minOverlap)
	}
	return nil
}

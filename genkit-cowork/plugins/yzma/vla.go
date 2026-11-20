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

package yzma

import (
	"context"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// BoundingBox describes image edges in thousandths of the image width or
// height. The origin is the top-left, x increases right and y increases down.
// Left and Top give the minimum corner; Right and Bottom give the maximum
// corner. The far edges of the image are at 1000, not 999.
type BoundingBox struct {
	Left   int `json:"left"`
	Top    int `json:"top"`
	Right  int `json:"right"`
	Bottom int `json:"bottom"`
}

// Validate rejects coordinates outside the image and boxes with zero or
// negative area. It cannot verify that the box matches the intended object.
func (b BoundingBox) Validate() error {
	if b.Left < 0 || b.Left >= b.Right || b.Right > 1000 ||
		b.Top < 0 || b.Top >= b.Bottom || b.Bottom > 1000 {
		return fmt.Errorf("yzma: invalid bounding box %+v: require 0 <= left < right <= 1000 and 0 <= top < bottom <= 1000", b)
	}
	return nil
}

// GenerateBoundingBox sends an inline JPEG or PNG and an instruction to a Yzma
// vision model with native constrained output. It returns one validated box,
// using the model's generation defaults. The prediction is data only; this
// function does not provide tools or execute any action based on the result.
func GenerateBoundingBox(ctx context.Context, g *genkit.Genkit, model ai.Model, image *ai.Part, instruction string) (*BoundingBox, error) {
	if model == nil {
		return nil, fmt.Errorf("yzma: bounding box prediction requires a vision model")
	}
	if image == nil || !image.IsMedia() || (image.ContentType != "image/jpeg" && image.ContentType != "image/png") {
		return nil, fmt.Errorf("yzma: bounding box prediction requires an inline JPEG or PNG")
	}
	if strings.TrimSpace(instruction) == "" {
		return nil, fmt.Errorf("yzma: bounding box prediction requires an instruction")
	}
	const coordinates = "\nReturn the box with integer edges left, top, right and bottom. " +
		"The top-left of the image is (0, 0) and the bottom-right is (1000, 1000). " +
		"Left must be smaller than right, and top must be smaller than bottom."
	box, _, err := genkit.GenerateData[BoundingBox](ctx, g, ai.WithModel(model),
		ai.WithMessages(ai.NewUserMessage(image, ai.NewTextPart(instruction+coordinates))))
	if err != nil {
		return nil, err
	}
	if box == nil {
		return nil, fmt.Errorf("yzma: generation did not return a bounding box")
	}
	if err := box.Validate(); err != nil {
		return nil, err
	}
	return box, nil
}

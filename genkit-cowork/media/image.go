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
	"context"
	"encoding/base64"
	"fmt"

	"github.com/firebase/genkit/go/ai"
)

type imageProcessor struct {
	mimeType string
}

func (p imageProcessor) Process(ctx context.Context, data []byte) ([]*ai.Document, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !isValidImageMimeType(p.mimeType) {
		return nil, errFileTypeNotSupported
	}

	resized, err := AutoResizeImage(data, p.mimeType)
	if err != nil {
		return nil, fmt.Errorf("process image document: %w", err)
	}

	dataURI := fmt.Sprintf("data:%s;base64,%s", resized.MimeType, base64.StdEncoding.EncodeToString(resized.Data))
	doc := &ai.Document{
		Content: []*ai.Part{ai.NewMediaPart(resized.MimeType, dataURI)},
		Metadata: map[string]any{
			"mimeType":         resized.MimeType,
			"originalMimeType": p.mimeType,
			"originalWidth":    resized.OriginalWidth,
			"originalHeight":   resized.OriginalHeight,
			"finalWidth":       resized.FinalWidth,
			"finalHeight":      resized.FinalHeight,
			"wasResized":       resized.WasResized,
		},
	}

	return []*ai.Document{doc}, nil
}

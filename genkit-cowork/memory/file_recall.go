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

package memory

import (
	"context"
	"fmt"
	"strings"
)

// SearchTenantFileChunks retrieves semantically similar tenant-global file
// chunks using the configured service indexer.
func SearchTenantFileChunks(ctx context.Context, service *FileIngestService, tenantID, query string, topK int) ([]FileChunkSearchResult, error) {
	if service == nil {
		return nil, fmt.Errorf("search tenant file chunks: service is nil")
	}
	if strings.TrimSpace(tenantID) == "" {
		return nil, fmt.Errorf("search tenant file chunks: tenantID is required")
	}
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("search tenant file chunks: query is required")
	}

	return service.SearchTenantFiles(ctx, FileChunkSearchInput{
		TenantID: tenantID,
		Query:    query,
		TopK:     topK,
	})
}

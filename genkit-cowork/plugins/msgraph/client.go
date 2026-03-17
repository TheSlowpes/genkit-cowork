package msgraph

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const graphBaseURL = "https://graph.microsoft.com/v1.0"

// GraphClient is the interface that abstracts all Microsoft Graph API calls
// made by the plugin's tools. Inject a custom implementation in tests or to
// swap out the transport layer.
type GraphClient interface {
	// GetMessages returns a page of email messages from the signed-in user's
	// mailbox. top limits the number of results (0 means use the server default).
	GetMessages(ctx context.Context, top int) ([]Message, error)

	// GetCalendarEvents returns upcoming calendar events for the signed-in
	// user. top limits the number of results (0 means use the server default).
	GetCalendarEvents(ctx context.Context, top int) ([]CalendarEvent, error)

	// GetDriveItem returns the metadata and text content of a OneDrive item
	// identified by itemPath (e.g. "/me/drive/root:/Documents/report.txt:").
	GetDriveItem(ctx context.Context, itemPath string) (*DriveItem, error)

	// UpdateDriveItem replaces the content of a OneDrive item identified by
	// itemPath with content.
	UpdateDriveItem(ctx context.Context, itemPath string, content []byte) error
}

// Message is a minimal representation of a Microsoft Graph mail message.
type Message struct {
	ID      string `json:"id"`
	Subject string `json:"subject"`
	From    struct {
		EmailAddress struct {
			Name    string `json:"name"`
			Address string `json:"address"`
		} `json:"emailAddress"`
	} `json:"from"`
	ReceivedDateTime string `json:"receivedDateTime"`
	BodyPreview      string `json:"bodyPreview"`
	IsRead           bool   `json:"isRead"`
}

// CalendarEvent is a minimal representation of a Microsoft Graph calendar event.
type CalendarEvent struct {
	ID      string `json:"id"`
	Subject string `json:"subject"`
	Start   struct {
		DateTime string `json:"dateTime"`
		TimeZone string `json:"timeZone"`
	} `json:"start"`
	End struct {
		DateTime string `json:"dateTime"`
		TimeZone string `json:"timeZone"`
	} `json:"end"`
	Location struct {
		DisplayName string `json:"displayName"`
	} `json:"location"`
	BodyPreview string `json:"bodyPreview"`
}

// DriveItem holds the metadata and text content of a OneDrive file.
type DriveItem struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Size    int64  `json:"size"`
	WebURL  string `json:"webUrl"`
	Content []byte `json:"-"`
}

// httpGraphClient is the default GraphClient implementation. It talks to the
// Microsoft Graph REST API using a Bearer access token.
type httpGraphClient struct {
	accessToken string
	httpClient  *http.Client
}

// newHTTPGraphClient creates a GraphClient backed by a simple Bearer-token
// HTTP client with a 30-second timeout.
func newHTTPGraphClient(accessToken string) GraphClient {
	return &httpGraphClient{
		accessToken: accessToken,
		httpClient:  &http.Client{Timeout: 30 * time.Second},
	}
}

// do executes an authenticated GET request against the Graph API and decodes
// the JSON response body into dst.
func (c *httpGraphClient) do(ctx context.Context, method, url string, body io.Reader, dst any) error {
	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return fmt.Errorf("msgraph: build request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.accessToken)
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/octet-stream")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("msgraph: request %s %s: %w", method, url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("msgraph: %s %s returned %s: %s", method, url, resp.Status, strings.TrimSpace(string(raw)))
	}

	if dst != nil {
		if err := json.NewDecoder(resp.Body).Decode(dst); err != nil {
			return fmt.Errorf("msgraph: decode response: %w", err)
		}
	}
	return nil
}

func (c *httpGraphClient) GetMessages(ctx context.Context, top int) ([]Message, error) {
	url := graphBaseURL + "/me/messages"
	if top > 0 {
		url = fmt.Sprintf("%s?$top=%d", url, top)
	}

	var result struct {
		Value []Message `json:"value"`
	}
	if err := c.do(ctx, http.MethodGet, url, nil, &result); err != nil {
		return nil, err
	}
	return result.Value, nil
}

func (c *httpGraphClient) GetCalendarEvents(ctx context.Context, top int) ([]CalendarEvent, error) {
	url := graphBaseURL + "/me/events"
	if top > 0 {
		url = fmt.Sprintf("%s?$top=%d", url, top)
	}

	var result struct {
		Value []CalendarEvent `json:"value"`
	}
	if err := c.do(ctx, http.MethodGet, url, nil, &result); err != nil {
		return nil, err
	}
	return result.Value, nil
}

func (c *httpGraphClient) GetDriveItem(ctx context.Context, itemPath string) (*DriveItem, error) {
	metaURL := graphBaseURL + itemPath
	var item DriveItem
	if err := c.do(ctx, http.MethodGet, metaURL, nil, &item); err != nil {
		return nil, err
	}

	contentURL := metaURL + "/content"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, contentURL, nil)
	if err != nil {
		return nil, fmt.Errorf("msgraph: build content request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.accessToken)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("msgraph: fetch content: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		raw, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("msgraph: fetch content returned %s: %s", resp.Status, strings.TrimSpace(string(raw)))
	}

	content, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("msgraph: read content body: %w", err)
	}
	item.Content = content
	return &item, nil
}

func (c *httpGraphClient) UpdateDriveItem(ctx context.Context, itemPath string, content []byte) error {
	url := graphBaseURL + itemPath + "/content"
	return c.do(ctx, http.MethodPut, url, strings.NewReader(string(content)), nil)
}

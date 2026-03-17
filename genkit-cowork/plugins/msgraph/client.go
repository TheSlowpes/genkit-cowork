package msgraph

import (
	"context"
	"fmt"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	msgraphsdkgo "github.com/microsoftgraph/msgraph-sdk-go"
	graphmodels "github.com/microsoftgraph/msgraph-sdk-go/models"
	graphusers "github.com/microsoftgraph/msgraph-sdk-go/users"
)

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

	// GetDriveItem returns the metadata and text content of a OneDrive item.
	// driveID is the ID of the drive and itemID is the item's unique ID within
	// that drive.
	GetDriveItem(ctx context.Context, driveID, itemID string) (*DriveItem, error)

	// UpdateDriveItem replaces the content of a OneDrive item identified by
	// driveID and itemID with content.
	UpdateDriveItem(ctx context.Context, driveID, itemID string, content []byte) error
}

// Message is a simplified representation of a Microsoft Graph mail message
// populated from the SDK's models.Messageable.
type Message struct {
	ID               string `json:"id"`
	Subject          string `json:"subject"`
	SenderName       string `json:"senderName"`
	SenderAddress    string `json:"senderAddress"`
	ReceivedDateTime string `json:"receivedDateTime"`
	BodyPreview      string `json:"bodyPreview"`
	IsRead           bool   `json:"isRead"`
}

// CalendarEvent is a simplified representation of a Microsoft Graph calendar
// event populated from the SDK's models.Eventable.
type CalendarEvent struct {
	ID               string `json:"id"`
	Subject          string `json:"subject"`
	StartDateTime    string `json:"startDateTime"`
	StartTimeZone    string `json:"startTimeZone"`
	EndDateTime      string `json:"endDateTime"`
	EndTimeZone      string `json:"endTimeZone"`
	LocationName     string `json:"locationName"`
	BodyPreview      string `json:"bodyPreview"`
}

// DriveItem holds simplified metadata and the text content of a OneDrive file,
// populated from the SDK's models.DriveItemable.
type DriveItem struct {
	ID      string `json:"id"`
	Name    string `json:"name"`
	Size    int64  `json:"size"`
	WebURL  string `json:"webUrl"`
	Content []byte `json:"-"`
}

// staticTokenCredential implements azcore.TokenCredential using a static
// access token string. This is suitable for short-lived OAuth 2.0 tokens
// obtained outside the SDK (e.g. from a web server auth flow).
type staticTokenCredential struct {
	token string
}

func (s *staticTokenCredential) GetToken(_ context.Context, _ policy.TokenRequestOptions) (azcore.AccessToken, error) {
	return azcore.AccessToken{
		Token:     s.token,
		ExpiresOn: time.Now().Add(time.Hour),
	}, nil
}

// newSDKGraphClient creates a GraphClient backed by the official
// microsoft/msgraph-sdk-go using the provided TokenCredential.
func newSDKGraphClient(cred azcore.TokenCredential) (GraphClient, error) {
	sdkClient, err := msgraphsdkgo.NewGraphServiceClientWithCredentials(cred, nil)
	if err != nil {
		return nil, fmt.Errorf("msgraph: create SDK client: %w", err)
	}
	return &sdkGraphClient{sdk: sdkClient}, nil
}

// sdkGraphClient is the default GraphClient implementation backed by the
// official microsoft/msgraph-sdk-go SDK client.
type sdkGraphClient struct {
	sdk *msgraphsdkgo.GraphServiceClient
}

func (c *sdkGraphClient) GetMessages(ctx context.Context, top int) ([]Message, error) {
	var cfg *graphusers.ItemMessagesRequestBuilderGetRequestConfiguration
	if top > 0 {
		top32 := int32(top)
		cfg = &graphusers.ItemMessagesRequestBuilderGetRequestConfiguration{
			QueryParameters: &graphusers.ItemMessagesRequestBuilderGetQueryParameters{
				Top: &top32,
			},
		}
	}

	resp, err := c.sdk.Me().Messages().Get(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("msgraph: get messages: %w", err)
	}

	items := resp.GetValue()
	msgs := make([]Message, 0, len(items))
	for _, m := range items {
		msgs = append(msgs, messageFromSDK(m))
	}
	return msgs, nil
}

func (c *sdkGraphClient) GetCalendarEvents(ctx context.Context, top int) ([]CalendarEvent, error) {
	var cfg *graphusers.ItemEventsRequestBuilderGetRequestConfiguration
	if top > 0 {
		top32 := int32(top)
		cfg = &graphusers.ItemEventsRequestBuilderGetRequestConfiguration{
			QueryParameters: &graphusers.ItemEventsRequestBuilderGetQueryParameters{
				Top: &top32,
			},
		}
	}

	resp, err := c.sdk.Me().Events().Get(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("msgraph: get events: %w", err)
	}

	items := resp.GetValue()
	events := make([]CalendarEvent, 0, len(items))
	for _, e := range items {
		events = append(events, calendarEventFromSDK(e))
	}
	return events, nil
}

func (c *sdkGraphClient) GetDriveItem(ctx context.Context, driveID, itemID string) (*DriveItem, error) {
	meta, err := c.sdk.Drives().ByDriveId(driveID).Items().ByDriveItemId(itemID).Get(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("msgraph: get drive item %s/%s: %w", driveID, itemID, err)
	}

	content, err := c.sdk.Drives().ByDriveId(driveID).Items().ByDriveItemId(itemID).Content().Get(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("msgraph: get drive item content %s/%s: %w", driveID, itemID, err)
	}

	item := driveItemFromSDK(meta)
	item.Content = content
	return &item, nil
}

func (c *sdkGraphClient) UpdateDriveItem(ctx context.Context, driveID, itemID string, content []byte) error {
	_, err := c.sdk.Drives().ByDriveId(driveID).Items().ByDriveItemId(itemID).Content().Put(ctx, content, nil)
	if err != nil {
		return fmt.Errorf("msgraph: update drive item %s/%s: %w", driveID, itemID, err)
	}
	return nil
}

// messageFromSDK maps a SDK models.Messageable to a Message.
func messageFromSDK(m graphmodels.Messageable) Message {
	msg := Message{}
	if v := m.GetId(); v != nil {
		msg.ID = *v
	}
	if v := m.GetSubject(); v != nil {
		msg.Subject = *v
	}
	if v := m.GetBodyPreview(); v != nil {
		msg.BodyPreview = *v
	}
	if v := m.GetIsRead(); v != nil {
		msg.IsRead = *v
	}
	if v := m.GetReceivedDateTime(); v != nil {
		msg.ReceivedDateTime = v.Format(time.RFC3339)
	}
	if f := m.GetFrom(); f != nil {
		if ea := f.GetEmailAddress(); ea != nil {
			if v := ea.GetName(); v != nil {
				msg.SenderName = *v
			}
			if v := ea.GetAddress(); v != nil {
				msg.SenderAddress = *v
			}
		}
	}
	return msg
}

// calendarEventFromSDK maps a SDK models.Eventable to a CalendarEvent.
func calendarEventFromSDK(e graphmodels.Eventable) CalendarEvent {
	ev := CalendarEvent{}
	if v := e.GetId(); v != nil {
		ev.ID = *v
	}
	if v := e.GetSubject(); v != nil {
		ev.Subject = *v
	}
	if v := e.GetBodyPreview(); v != nil {
		ev.BodyPreview = *v
	}
	if s := e.GetStart(); s != nil {
		if v := s.GetDateTime(); v != nil {
			ev.StartDateTime = *v
		}
		if v := s.GetTimeZone(); v != nil {
			ev.StartTimeZone = *v
		}
	}
	if en := e.GetEnd(); en != nil {
		if v := en.GetDateTime(); v != nil {
			ev.EndDateTime = *v
		}
		if v := en.GetTimeZone(); v != nil {
			ev.EndTimeZone = *v
		}
	}
	if l := e.GetLocation(); l != nil {
		if v := l.GetDisplayName(); v != nil {
			ev.LocationName = *v
		}
	}
	return ev
}

// driveItemFromSDK maps a SDK models.DriveItemable to a DriveItem.
func driveItemFromSDK(d graphmodels.DriveItemable) DriveItem {
	item := DriveItem{}
	if v := d.GetId(); v != nil {
		item.ID = *v
	}
	if v := d.GetName(); v != nil {
		item.Name = *v
	}
	if v := d.GetSize(); v != nil {
		item.Size = *v
	}
	if v := d.GetWebUrl(); v != nil {
		item.WebURL = *v
	}
	return item
}

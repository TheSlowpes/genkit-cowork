package msgraph

import (
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// listEmailsInput is the input schema for the list-emails tool.
type listEmailsInput struct {
	// Top is the maximum number of messages to return. Zero uses the server
	// default (typically 10).
	Top int `json:"top,omitempty" jsonschema_description:"Maximum number of emails to return (0 = server default)."`
}

// listCalendarEventsInput is the input schema for the list-calendar-events tool.
type listCalendarEventsInput struct {
	// Top is the maximum number of events to return. Zero uses the server
	// default (typically 10).
	Top int `json:"top,omitempty" jsonschema_description:"Maximum number of calendar events to return (0 = server default)."`
}

// readOneDriveFileInput is the input schema for the read-onedrive-file tool.
type readOneDriveFileInput struct {
	// ItemPath is the Graph API path to the file, e.g.
	// "/me/drive/root:/Documents/report.txt:".
	ItemPath string `json:"itemPath" jsonschema_description:"Graph API path to the OneDrive file, e.g. /me/drive/root:/Documents/report.txt:"`
}

// editOneDriveFileInput is the input schema for the edit-onedrive-file tool.
type editOneDriveFileInput struct {
	// ItemPath is the Graph API path to the file, e.g.
	// "/me/drive/root:/Documents/report.txt:".
	ItemPath string `json:"itemPath" jsonschema_description:"Graph API path to the OneDrive file to overwrite."`

	// Content is the new UTF-8 text content to write to the file.
	Content string `json:"content" jsonschema_description:"New text content to write to the file."`
}

// ListEmailsTool returns a Genkit tool that lists emails from the signed-in
// user's mailbox using the Microsoft Graph /me/messages endpoint.
func ListEmailsTool(g *genkit.Genkit, client GraphClient) ai.Tool {
	return genkit.DefineTool(
		g,
		"list-emails",
		"List emails from the signed-in user's mailbox. Returns subject, sender, received time and a body preview for each message.",
		func(ctx *ai.ToolContext, input listEmailsInput) ([]Message, error) {
			messages, err := client.GetMessages(ctx, input.Top)
			if err != nil {
				return nil, fmt.Errorf("list-emails: %w", err)
			}
			return messages, nil
		},
	)
}

// ListCalendarEventsTool returns a Genkit tool that lists upcoming calendar
// events for the signed-in user using the Microsoft Graph /me/events endpoint.
func ListCalendarEventsTool(g *genkit.Genkit, client GraphClient) ai.Tool {
	return genkit.DefineTool(
		g,
		"list-calendar-events",
		"List upcoming calendar events for the signed-in user. Returns subject, start/end times, location, and a body preview.",
		func(ctx *ai.ToolContext, input listCalendarEventsInput) ([]CalendarEvent, error) {
			events, err := client.GetCalendarEvents(ctx, input.Top)
			if err != nil {
				return nil, fmt.Errorf("list-calendar-events: %w", err)
			}
			return events, nil
		},
	)
}

// ReadOneDriveFileTool returns a Genkit tool that retrieves the metadata and
// text content of a OneDrive file via the Microsoft Graph API.
func ReadOneDriveFileTool(g *genkit.Genkit, client GraphClient) ai.Tool {
	return genkit.DefineTool(
		g,
		"read-onedrive-file",
		"Read a file from OneDrive. Returns the file metadata (name, size, web URL) and its text content.",
		func(ctx *ai.ToolContext, input readOneDriveFileInput) (*DriveItem, error) {
			if input.ItemPath == "" {
				return nil, fmt.Errorf("read-onedrive-file: itemPath is required")
			}
			item, err := client.GetDriveItem(ctx, input.ItemPath)
			if err != nil {
				return nil, fmt.Errorf("read-onedrive-file: %w", err)
			}
			return item, nil
		},
	)
}

// EditOneDriveFileTool returns a Genkit tool that replaces the content of a
// OneDrive file with new text via the Microsoft Graph API.
func EditOneDriveFileTool(g *genkit.Genkit, client GraphClient) ai.Tool {
	return genkit.DefineTool(
		g,
		"edit-onedrive-file",
		"Overwrite a OneDrive file with new text content.",
		func(ctx *ai.ToolContext, input editOneDriveFileInput) (string, error) {
			if input.ItemPath == "" {
				return "", fmt.Errorf("edit-onedrive-file: itemPath is required")
			}
			if err := client.UpdateDriveItem(ctx, input.ItemPath, []byte(input.Content)); err != nil {
				return "", fmt.Errorf("edit-onedrive-file: %w", err)
			}
			return fmt.Sprintf("successfully updated %s", input.ItemPath), nil
		},
	)
}

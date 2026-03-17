// Package msgraph provides a Microsoft Graph API plugin for Firebase Genkit.
//
// The plugin registers tools that allow AI agents to interact with Microsoft
// 365 services on behalf of a signed-in user:
//
//   - list-emails         — list messages from the user's mailbox
//   - list-calendar-events — list upcoming calendar events
//   - read-onedrive-file  — read a file from OneDrive
//   - edit-onedrive-file  — overwrite a file in OneDrive
//
// # Quick start
//
//	plugin := &msgraph.MSGraph{AccessToken: os.Getenv("MSGRAPH_ACCESS_TOKEN")}
//	g, _ := genkit.Init(ctx, genkit.WithPlugins(plugin))
//	tools := plugin.Tools(g)
//
// The AccessToken must be a valid OAuth 2.0 Bearer token with the required
// Microsoft Graph delegated permissions:
//
//   - Mail.Read
//   - Calendars.Read
//   - Files.ReadWrite
//
// For custom authentication or testing, provide a ClientFactory that constructs
// an alternative GraphClient implementation.
package msgraph

import (
	"context"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

const provider = "msgraph"

var _ api.Plugin = (*MSGraph)(nil)

// ClientFactory is a function that constructs a GraphClient from an access
// token. Provide a custom factory via MSGraph.ClientFactory to swap the
// default HTTP implementation for testing or alternative transports.
type ClientFactory func(accessToken string) GraphClient

// MSGraph is the configuration for the Microsoft Graph plugin.
// Pass it to genkit.WithPlugins() to register it with a Genkit application.
type MSGraph struct {
	// AccessToken is the OAuth 2.0 Bearer token used to authenticate all
	// Microsoft Graph API requests. The token must have the delegated
	// permissions required by the enabled tools (Mail.Read, Calendars.Read,
	// Files.ReadWrite).
	//
	// Required unless ClientFactory is provided and constructs a client
	// without needing an access token.
	AccessToken string

	// ClientFactory is an optional factory function for constructing the
	// GraphClient. When nil, the default HTTP implementation is used.
	// Inject a custom factory in tests to avoid real network calls.
	ClientFactory ClientFactory

	// client is the initialized GraphClient, set during Init.
	client GraphClient

	mu      sync.Mutex
	initted bool
}

// Name implements api.Plugin.
// Returns the unique provider identifier used by Genkit's registry.
func (m *MSGraph) Name() string {
	return provider
}

// Init implements api.Plugin.
// Validates the plugin configuration and initializes the underlying
// GraphClient. Panics if called more than once or if AccessToken is empty and
// no ClientFactory is set.
//
// Init does not register any Genkit actions; tools are registered separately
// by the caller via Tools().
func (m *MSGraph) Init(_ context.Context) []api.Action {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.initted {
		panic("MSGraph plugin Init called more than once")
	}
	m.initted = true

	if m.ClientFactory != nil {
		m.client = m.ClientFactory(m.AccessToken)
		return nil
	}

	if m.AccessToken == "" {
		panic("MSGraph plugin: AccessToken is required when ClientFactory is not set")
	}

	m.client = newHTTPGraphClient(m.AccessToken)
	return nil
}

// Client returns the GraphClient initialised by Init.
// Panics if called before Init.
func (m *MSGraph) Client() GraphClient {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.initted {
		panic("MSGraph plugin: Client() called before Init()")
	}
	return m.client
}

// Tools registers and returns all Microsoft Graph tools with the provided
// Genkit instance. Call this after Init.
//
//   - list-emails
//   - list-calendar-events
//   - read-onedrive-file
//   - edit-onedrive-file
func (m *MSGraph) Tools(g *genkit.Genkit) []ai.Tool {
	client := m.Client()
	return []ai.Tool{
		ListEmailsTool(g, client),
		ListCalendarEventsTool(g, client),
		ReadOneDriveFileTool(g, client),
		EditOneDriveFileTool(g, client),
	}
}

// Package msgraph provides a Microsoft Graph API plugin for Firebase Genkit.
//
// The plugin registers tools that allow AI agents to interact with Microsoft
// 365 services on behalf of a signed-in user:
//
//   - list-emails          — list messages from the user's mailbox
//   - list-calendar-events — list upcoming calendar events
//   - read-onedrive-file   — read a file from OneDrive
//   - edit-onedrive-file   — overwrite a file in OneDrive
//
// # Quick start
//
//	plugin := &msgraph.MSGraph{AccessToken: os.Getenv("MSGRAPH_ACCESS_TOKEN")}
//	g, _ := genkit.Init(ctx, genkit.WithPlugins(plugin))
//	tools := plugin.Tools(g)
//
// AccessToken must be a valid OAuth 2.0 Bearer token with the required
// Microsoft Graph delegated permissions:
//
//   - Mail.Read
//   - Calendars.Read
//   - Files.ReadWrite
//
// For production workloads that use managed identity or application credentials,
// supply a Credential (azcore.TokenCredential) instead of AccessToken. The
// credential is forwarded to NewGraphServiceClientWithCredentials from the
// official microsoft/msgraph-sdk-go SDK.
//
// For testing, supply a ClientFactory that constructs a mock GraphClient
// without making real API calls.
package msgraph

import (
	"context"
	"sync"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

const provider = "msgraph"

var _ api.Plugin = (*MSGraph)(nil)

// ClientFactory is a function that constructs a GraphClient from an access
// token string. Provide a custom factory via MSGraph.ClientFactory to inject
// a mock implementation in tests without making real API calls.
type ClientFactory func(accessToken string) GraphClient

// MSGraph is the configuration for the Microsoft Graph plugin.
// Pass it to genkit.WithPlugins() to register it with a Genkit application.
type MSGraph struct {
	// AccessToken is a short-lived OAuth 2.0 Bearer token used to authenticate
	// Microsoft Graph API requests. The token must carry the delegated
	// permissions required by the enabled tools (Mail.Read, Calendars.Read,
	// Files.ReadWrite).
	//
	// Use AccessToken for simple token-based authentication. For production
	// workloads that require automatic token refresh, supply Credential instead.
	//
	// Either AccessToken or Credential must be set when ClientFactory is nil.
	AccessToken string

	// Credential is an azcore.TokenCredential that the SDK uses to obtain and
	// refresh access tokens automatically. When set, Credential takes precedence
	// over AccessToken.
	//
	// Typical choices include azidentity.ClientSecretCredential,
	// azidentity.ManagedIdentityCredential, or azidentity.InteractiveBrowserCredential.
	// For testing, azcore/fake.TokenCredential can be used.
	Credential azcore.TokenCredential

	// ClientFactory is an optional factory function for constructing the
	// GraphClient. When set, ClientFactory is called with AccessToken in Init
	// and takes full precedence over both AccessToken and Credential.
	// Use this to inject a mock GraphClient in tests.
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
// Creates the underlying GraphClient using the official msgraph-sdk-go SDK.
// The SDK client is created with NewGraphServiceClientWithCredentials:
//   - If ClientFactory is set, it is called to construct the client instead.
//   - If Credential is set, it is passed directly to the SDK.
//   - If only AccessToken is set, it is wrapped in a staticTokenCredential.
//
// Panics if Init is called more than once, or if neither AccessToken,
// Credential, nor ClientFactory is provided.
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

	cred := m.Credential
	if cred == nil {
		if m.AccessToken == "" {
			panic("MSGraph plugin: one of AccessToken, Credential, or ClientFactory must be set")
		}
		cred = &staticTokenCredential{token: m.AccessToken}
	}

	client, err := newSDKGraphClient(cred)
	if err != nil {
		panic("MSGraph plugin: " + err.Error())
	}
	m.client = client
	return nil
}

// Client returns the GraphClient initialized by Init.
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

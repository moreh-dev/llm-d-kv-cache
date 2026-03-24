/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package types

import (
	"encoding/json"
	"errors"
	"strings"
)

// Conversation represents a single message in a conversation.
type Conversation struct {
	// Role is the message Role, optional values are 'user', 'assistant', ...
	Role string `json:"role,omitempty"`
	// Content defines text of this message
	Content Content `json:"content,omitempty"`
}

type Content struct {
	Raw        string
	Structured []ContentBlock
}

type ContentBlock struct {
	Type       string     `json:"type"`
	Text       string     `json:"text,omitempty"`
	ImageURL   ImageBlock `json:"image_url,omitempty"`
	InputAudio AudioBlock `json:"input_audio,omitempty"`
	VideoURL   VideoBlock `json:"video_url,omitempty"`
}

type ImageBlock struct {
	Url string `json:"url,omitempty"`
}

type AudioBlock struct {
	Data   string `json:"data,omitempty"`
	Format string `json:"format,omitempty"`
}

type VideoBlock struct {
	Url string `json:"url,omitempty"`
}

// UnmarshalJSON allow use both format.
func (mc *Content) UnmarshalJSON(data []byte) error {
	// Raw format
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		mc.Raw = str
		return nil
	}

	// Block format
	var blocks []ContentBlock
	if err := json.Unmarshal(data, &blocks); err == nil {
		mc.Structured = blocks
		return nil
	}

	return errors.New("content format not supported")
}

func (mc Content) MarshalJSON() ([]byte, error) {
	if mc.Raw != "" {
		return json.Marshal(mc.Raw)
	}
	if mc.Structured != nil {
		return json.Marshal(mc.Structured)
	}
	return json.Marshal("")
}

func (mc Content) PlainText() string {
	if mc.Raw != "" {
		return mc.Raw
	}
	var sb strings.Builder
	for _, block := range mc.Structured {
		if block.Type == "text" {
			sb.WriteString(block.Text)
			sb.WriteString(" ")
		}
	}
	return sb.String()
}

// RenderChatRequest represents the request to render a chat template.
type RenderChatRequest struct {
	// The Python wrapper will handle converting this to a batched list if needed.
	Conversation              []Conversation         `json:"conversation"`
	Tools                     []interface{}          `json:"tools,omitempty"`
	Documents                 []interface{}          `json:"documents,omitempty"`
	ChatTemplate              string                 `json:"chat_template,omitempty"`
	ReturnAssistantTokensMask bool                   `json:"return_assistant_tokens_mask,omitempty"`
	ContinueFinalMessage      bool                   `json:"continue_final_message,omitempty"`
	AddGenerationPrompt       bool                   `json:"add_generation_prompt,omitempty"`
	TruncatePromptTokens      *int                   `json:"truncate_prompt_tokens,omitempty"`
	ChatTemplateKWArgs        map[string]interface{} `json:"chat_template_kwargs,omitempty"`
}

type RenderRequest struct {
	Text             string `json:"text"`
	AddSpecialTokens bool   `json:"add_special_tokens,omitempty"`
}

// Offset represents a character offset range with [start, end] indices.
type Offset [2]uint

type RenderResponse struct {
	TokenIDs       []uint32 `json:"input_ids"`
	OffsetMappings []Offset `json:"offset_mapping"`
}

// DeepCopy creates a deep copy of the RenderChatRequest.
func (req *RenderChatRequest) DeepCopy() (*RenderChatRequest, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	var out RenderChatRequest
	err = json.Unmarshal(b, &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}

// DeepCopy creates a deep copy of the RenderRequest.
func (req *RenderRequest) DeepCopy() (*RenderRequest, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	var out RenderRequest
	err = json.Unmarshal(b, &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}

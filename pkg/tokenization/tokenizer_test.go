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

//nolint:testpackage // need to test internal types
package tokenization

import (
	"encoding/json"
	"testing"
)

func TestTokenizerOptions_ToolCallFieldsUnmarshal(t *testing.T) {
	var o TokenizerOptions
	if err := json.Unmarshal([]byte(`{"enableAutoToolChoice":true,"toolCallParser":"gemma4"}`), &o); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !o.EnableAutoToolChoice {
		t.Errorf("EnableAutoToolChoice = false, want true")
	}
	if o.ToolCallParser != "gemma4" {
		t.Errorf("ToolCallParser = %q, want \"gemma4\"", o.ToolCallParser)
	}
}

func TestDefaultTokenizerOptions_ToolCallDefaults(t *testing.T) {
	d := DefaultTokenizerOptions()
	if d.EnableAutoToolChoice {
		t.Errorf("default EnableAutoToolChoice = true, want false")
	}
	if d.ToolCallParser != "" {
		t.Errorf("default ToolCallParser = %q, want empty", d.ToolCallParser)
	}
}

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"
)

const (
	exitOK = iota
	exitUsage
	exitNoAPIKey
	exitAPI
	exitInput
)

const requestTimeout = 2 * time.Minute

var docs = `
osai - a one-shot command-line client for the OpenAI Responses API

Usage:
  osai -model MODEL [options]

Options:
  -model string
        Model to use (required)

  -provider string
        API provider: openai or xai (default "openai")

  -system string
        System prompt or @file

  -prompt string
        User prompt or @file

  -input file
        Read additional input from file

  -state file
        Read and update the previous response ID from file.
        The parent directory must already exist.

  -temperature float
        Sampling temperature

  -max-output-tokens int
        Maximum output tokens

Input:
  User input may be supplied using -prompt, stdin, or -input.

  If both -prompt and stdin/file are present, they are concatenated
  with a blank line between them.

Environment:
  OPENAI_API_KEY  Used with -provider openai
  XAI_API_KEY     Used with -provider xai
`

func usage() {
	fmt.Fprintf(os.Stderr, "%s\n", docs)
}

func fail(code int, err error) {
	fmt.Fprintln(os.Stderr, "osai:", err)
	os.Exit(code)
}

func usageError(err error) {
	fail(exitUsage, fmt.Errorf("%v\n\nTry 'osai -h' for more information.", err))
}

func expandAtFile(value string) (string, error) {
	if !strings.HasPrefix(value, "@") {
		return value, nil
	}

	path := value[1:]

	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}

		path = filepath.Join(home, path[2:])
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read %q: %w", path, err)
	}

	return string(data), nil
}

func isStdinPipe() bool {
	info, err := os.Stdin.Stat()
	if err != nil {
		return false
	}

	return (info.Mode() & os.ModeCharDevice) == 0
}

func readInput(inputFile string) (string, error) {
	stdin := isStdinPipe()

	if inputFile != "" && stdin {
		return "", errors.New("-input and stdin cannot both be used")
	}

	switch {
	case inputFile != "":
		data, err := os.ReadFile(inputFile)
		if err != nil {
			return "", fmt.Errorf("read %q: %w", inputFile, err)
		}
		return string(data), nil

	case stdin:
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			return "", err
		}
		return string(data), nil

	default:
		return "", nil
	}
}

func buildPrompt(prompt, input string) string {
	switch {
	case prompt == "":
		return input
	case input == "":
		return prompt
	default:
		return prompt + "\n\n" + input
	}
}

func buildRequest(
	provider string,
	model string,
	system string,
	prompt string,
	previousResponseID string,
	maxOutputTokens int,
	temperature float64,
) responses.ResponseNewParams {
	req := responses.ResponseNewParams{
		Model: model,
	}

	if provider == "xai" {
		input := make([]responses.ResponseInputItemUnionParam, 0, 2)

		if previousResponseID == "" && system != "" {
			input = append(input,
				responses.ResponseInputItemParamOfMessage(
					system,
					responses.EasyInputMessageRoleSystem,
				),
			)
		}

		input = append(input,
			responses.ResponseInputItemParamOfMessage(
				prompt,
				responses.EasyInputMessageRoleUser,
			),
		)

		req.Input = responses.ResponseNewParamsInputUnion{
			OfInputItemList: input,
		}
	} else {
		req.Input = responses.ResponseNewParamsInputUnion{
			OfString: openai.String(prompt),
		}

		if system != "" {
			req.Instructions = openai.String(system)
		}
	}

	if previousResponseID != "" {
		req.PreviousResponseID = openai.String(previousResponseID)
	}

	if maxOutputTokens > 0 {
		req.MaxOutputTokens = openai.Int(int64(maxOutputTokens))
	}

	if temperature >= 0 {
		req.Temperature = openai.Float(temperature)
	}

	return req
}

func readState(path string) (string, error) {
	if path == "" {
		return "", nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", fmt.Errorf("read %q: %w", path, err)
	}

	id := strings.TrimSpace(string(data))
	if id == "" {
		return "", fmt.Errorf("state file %q is empty", path)
	}

	return id, nil
}

func writeState(path, id string) error {
	if path == "" {
		return nil
	}

	if err := os.WriteFile(path, []byte(id+"\n"), 0600); err != nil {
		return fmt.Errorf("write %q: %w", path, err)
	}

	return nil
}

func main() {
	var (
		model           string
		provider        string
		system          string
		prompt          string
		inputFile       string
		stateFile       string
		maxOutputTokens int
		temperature     float64
	)

	flag.StringVar(&model, "model", "", "Model to use (required)")
	flag.StringVar(&provider, "provider", "openai", "API provider: openai or xai")
	flag.StringVar(&system, "system", "", "System prompt or @file")
	flag.StringVar(&prompt, "prompt", "", "User prompt or @file")
	flag.StringVar(&inputFile, "input", "", "Read additional input from file")
	flag.StringVar(&stateFile, "state", "", "Read and update the previous response ID from file")
	flag.IntVar(&maxOutputTokens, "max-output-tokens", 0, "Maximum output tokens")
	flag.Float64Var(&temperature, "temperature", -1, "Sampling temperature")

	flag.Usage = usage
	flag.Parse()

	if model == "" {
		usageError(errors.New("missing required -model"))
	}

	var apiKeyEnv string

	switch provider {
	case "openai":
		apiKeyEnv = "OPENAI_API_KEY"
	case "xai":
		apiKeyEnv = "XAI_API_KEY"
	default:
		usageError(fmt.Errorf(
			"unsupported provider %q (expected openai or xai)",
			provider,
		))
	}

	apiKey := os.Getenv(apiKeyEnv)
	if apiKey == "" {
		fail(exitNoAPIKey, fmt.Errorf("%s is not set", apiKeyEnv))
	}

	systemText, err := expandAtFile(system)
	if err != nil {
		fail(exitInput, err)
	}

	promptText, err := expandAtFile(prompt)
	if err != nil {
		fail(exitInput, err)
	}

	userInput, err := readInput(inputFile)
	if err != nil {
		fail(exitInput, err)
	}

	finalPrompt := buildPrompt(promptText, userInput)
	if finalPrompt == "" {
		usageError(errors.New("no input provided"))
	}

	previousResponseID, err := readState(stateFile)
	if err != nil {
		fail(exitInput, err)
	}

	clientOptions := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}

	if provider == "xai" {
		clientOptions = append(
			clientOptions,
			option.WithBaseURL("https://api.x.ai/v1"),
		)
	}

	client := openai.NewClient(clientOptions...)

	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	defer cancel()

	req := buildRequest(
		provider,
		model,
		systemText,
		finalPrompt,
		previousResponseID,
		maxOutputTokens,
		temperature,
	)

	resp, err := client.Responses.New(ctx, req)
	if err != nil {
		fail(exitAPI, err)
	}

	text := resp.OutputText()

	fmt.Print(text)

	if !strings.HasSuffix(text, "\n") {
		fmt.Println()
	}

	if err := writeState(stateFile, resp.ID); err != nil {
		fail(exitInput, err)
	}
}

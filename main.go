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

  -system string
        System prompt or @file

  -prompt string
        User prompt or @file

  -input-file file
        Read additional input from file

  -temperature float
        Sampling temperature

  -max-output-tokens int
        Maximum output tokens

Input:
  User input may be supplied using -prompt, stdin, or -input-file.

  If both -prompt and stdin/file are present, they are concatenated
  with a blank line between them.

Environment:
  OPENAI_API_KEY
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
		return "", errors.New("-input-file and stdin cannot both be used")
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
	model string,
	system string,
	prompt string,
	maxOutputTokens int,
	temperature float64,
) responses.ResponseNewParams {
	req := responses.ResponseNewParams{
		Model: model,
		Input: responses.ResponseNewParamsInputUnion{
			OfString: openai.String(prompt),
		},
	}

	if system != "" {
		req.Instructions = openai.String(system)
	}

	if maxOutputTokens > 0 {
		req.MaxOutputTokens = openai.Int(int64(maxOutputTokens))
	}

	if temperature != 0 {
		req.Temperature = openai.Float(temperature)
	}

	return req
}

func main() {
	var (
		model           string
		system          string
		prompt          string
		inputFile       string
		maxOutputTokens int
		temperature     float64
	)

	flag.StringVar(&model, "model", "", "Model to use (required)")
	flag.StringVar(&system, "system", "", "System prompt or @file")
	flag.StringVar(&prompt, "prompt", "", "User prompt or @file")
	flag.StringVar(&inputFile, "input-file", "", "Read additional input from file")
	flag.IntVar(&maxOutputTokens, "max-output-tokens", 0, "Maximum output tokens")
	flag.Float64Var(&temperature, "temperature", 0, "Sampling temperature")

	flag.Usage = usage
	flag.Parse()

	if model == "" {
		usageError(errors.New("missing required -model"))
	}

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fail(exitNoAPIKey, errors.New("OPENAI_API_KEY is not set"))
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

	client := openai.NewClient()

	ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
	defer cancel()

	req := buildRequest(
		model,
		systemText,
		finalPrompt,
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
}

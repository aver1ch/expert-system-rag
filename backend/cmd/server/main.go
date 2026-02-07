package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"time"
)

type analyzeRequest struct {
	Text string `json:"text"`
}

type errorItem struct {
	Category string  `json:"category"`
	Message  string  `json:"message"`
	Location *string `json:"location,omitempty"`
	Source   *string `json:"source,omitempty"`
}

type analyzeResponse struct {
	Errors []errorItem `json:"errors"`
}

func coreURL() string {
	if v := os.Getenv("CORE_SERVICE_URL"); v != "" {
		return v
	}
	// По умолчанию считаем, что core-сервис запущен локально на 8000 порту
	return "http://localhost:8000"
}

func analyzeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	defer r.Body.Close()

	var req analyzeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}

	if req.Text == "" {
		http.Error(w, "field 'text' must not be empty", http.StatusBadRequest)
		return
	}

	bodyBytes, err := json.Marshal(req)
	if err != nil {
		http.Error(w, "failed to serialize request", http.StatusInternalServerError)
		return
	}

	client := &http.Client{
		Timeout: 120 * time.Second,
	}

	coreEndpoint := coreURL() + "/analyze"
	coreReq, err := http.NewRequest(http.MethodPost, coreEndpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		http.Error(w, "failed to build request to core service", http.StatusInternalServerError)
		return
	}

	coreReq.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(coreReq)
	if err != nil {
		log.Printf("failed to call core service: %v", err)
		http.Error(w, "core service is unavailable", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)

	if _, err := io.Copy(w, resp.Body); err != nil {
		log.Printf("failed to proxy response body: %v", err)
	}
}

func waitForCoreService() {
	client := &http.Client{
		Timeout: 5 * time.Second,
	}

	coreEndpoint := coreURL() + "/analyze"

	for {
		req, err := http.NewRequest(http.MethodPost, coreEndpoint, bytes.NewReader([]byte(`{"text": "health check"}`)))
		if err != nil {
			log.Printf("failed to build health check request: %v", err)
			time.Sleep(2 * time.Second)
			continue
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err == nil && resp.StatusCode == http.StatusOK {
			resp.Body.Close()
			log.Printf("Core service is ready at %s", coreEndpoint)
			return
		}

		if resp != nil {
			resp.Body.Close()
		}

		log.Printf("Waiting for core service at %s...", coreEndpoint)
		time.Sleep(2 * time.Second)
	}
}

func main() {
	log.Printf("Waiting for core service...")
	waitForCoreService()

	http.HandleFunc("/analyze", analyzeHandler)

	addr := ":8080"
	log.Printf("Backend server is listening on %s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

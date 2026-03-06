package main

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type analyzeRequest struct {
	Text  string `json:"text"`
	Title string `json:"title,omitempty"`
}

type errorItem struct {
	Category    string  `json:"category"`
	Message     string  `json:"message"`
	Location    *string `json:"location,omitempty"`
	Source      *string `json:"source,omitempty"`
	Suggestion  *string `json:"suggestion,omitempty"`
	Replacement *string `json:"replacement,omitempty"`
}

type analyzeSummary struct {
	ExactDuplicatePercent   float64 `json:"exact_duplicate_percent"`
	PartialDuplicatePercent float64 `json:"partial_duplicate_percent"`
}

type analyzeResponse struct {
	Errors  []errorItem    `json:"errors"`
	Summary analyzeSummary `json:"summary"`
}

type documentMeta struct {
	ID            string    `json:"id"`
	Name          string    `json:"name"`
	OriginalExt   string    `json:"original_ext"`
	OriginalPath  string    `json:"original_path"`
	CurrentText   string    `json:"current_text"`
	LastUpdatedAt time.Time `json:"last_updated_at"`
}

type documentStore struct {
	mu       sync.RWMutex
	baseDir  string
	metaPath string
	docs     map[string]documentMeta
}

func coreURL() string {
	if v := os.Getenv("CORE_SERVICE_URL"); v != "" {
		return v
	}
	return "http://localhost:8000"
}

func analyzeTimeout() time.Duration {
	raw := strings.TrimSpace(os.Getenv("BACKEND_ANALYZE_TIMEOUT_SEC"))
	if raw == "" {
		return 0
	}
	sec, err := strconv.Atoi(raw)
	if err != nil || sec < 0 {
		return 0
	}
	if sec == 0 {
		return 0
	}
	return time.Duration(sec) * time.Second
}

func newDocumentStore(baseDir string) (*documentStore, error) {
	if err := os.MkdirAll(baseDir, 0o755); err != nil {
		return nil, err
	}
	s := &documentStore{
		baseDir:  baseDir,
		metaPath: filepath.Join(baseDir, "meta.json"),
		docs:     make(map[string]documentMeta),
	}
	if err := s.load(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *documentStore) load() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := os.ReadFile(s.metaPath)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}

	var list []documentMeta
	if err := json.Unmarshal(data, &list); err != nil {
		return err
	}

	for _, item := range list {
		s.docs[item.ID] = item
	}
	return nil
}

func (s *documentStore) saveLocked() error {
	list := make([]documentMeta, 0, len(s.docs))
	for _, item := range s.docs {
		list = append(list, item)
	}
	sort.Slice(list, func(i, j int) bool {
		return list[i].LastUpdatedAt.After(list[j].LastUpdatedAt)
	})

	data, err := json.MarshalIndent(list, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.metaPath, data, 0o644)
}

func (s *documentStore) list() []documentMeta {
	s.mu.RLock()
	defer s.mu.RUnlock()

	list := make([]documentMeta, 0, len(s.docs))
	for _, item := range s.docs {
		list = append(list, item)
	}
	sort.Slice(list, func(i, j int) bool {
		return list[i].LastUpdatedAt.After(list[j].LastUpdatedAt)
	})
	return list
}

func randomID() (string, error) {
	buf := make([]byte, 8)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return hex.EncodeToString(buf), nil
}

func requestIDFrom(r *http.Request) string {
	existing := strings.TrimSpace(r.Header.Get("X-Request-ID"))
	if existing != "" {
		return existing
	}
	id, err := randomID()
	if err != nil {
		return fmt.Sprintf("req-%d", time.Now().UTC().UnixNano())
	}
	return "req-" + id
}

func saveUploadedFile(baseDir, docID string, file multipart.File, header *multipart.FileHeader) (string, string, error) {
	ext := strings.ToLower(filepath.Ext(header.Filename))
	if ext != ".pdf" && ext != ".docx" {
		return "", "", fmt.Errorf("unsupported extension: %s", ext)
	}

	docDir := filepath.Join(baseDir, docID)
	if err := os.MkdirAll(docDir, 0o755); err != nil {
		return "", "", err
	}

	storedPath := filepath.Join(docDir, "original"+ext)
	out, err := os.Create(storedPath)
	if err != nil {
		return "", "", err
	}
	defer out.Close()

	if _, err := io.Copy(out, file); err != nil {
		return "", "", err
	}
	return storedPath, ext, nil
}

func (s *documentStore) create(name, text string, file multipart.File, header *multipart.FileHeader) (documentMeta, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	id, err := randomID()
	if err != nil {
		return documentMeta{}, err
	}

	path, ext, err := saveUploadedFile(s.baseDir, id, file, header)
	if err != nil {
		return documentMeta{}, err
	}

	item := documentMeta{
		ID:            id,
		Name:          name,
		OriginalExt:   ext,
		OriginalPath:  path,
		CurrentText:   text,
		LastUpdatedAt: time.Now().UTC(),
	}
	s.docs[id] = item
	if err := s.saveLocked(); err != nil {
		return documentMeta{}, err
	}
	return item, nil
}

func (s *documentStore) get(id string) (documentMeta, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	item, ok := s.docs[id]
	return item, ok
}

func (s *documentStore) update(id, text string) (documentMeta, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	item, ok := s.docs[id]
	if !ok {
		return documentMeta{}, os.ErrNotExist
	}
	item.CurrentText = text
	item.LastUpdatedAt = time.Now().UTC()
	s.docs[id] = item
	if err := s.saveLocked(); err != nil {
		return documentMeta{}, err
	}
	return item, nil
}

func (s *documentStore) remove(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	item, ok := s.docs[id]
	if !ok {
		return os.ErrNotExist
	}
	delete(s.docs, id)
	if err := os.RemoveAll(filepath.Dir(item.OriginalPath)); err != nil {
		return err
	}
	return s.saveLocked()
}

func analyzeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	startedAt := time.Now()

	defer r.Body.Close()

	var req analyzeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}

	if strings.TrimSpace(req.Text) == "" {
		http.Error(w, "field 'text' must not be empty", http.StatusBadRequest)
		return
	}
	reqID := requestIDFrom(r)
	log.Printf("[analyze][%s] started text_len=%d", reqID, len(strings.TrimSpace(req.Text)))

	bodyBytes, err := json.Marshal(req)
	if err != nil {
		http.Error(w, "failed to serialize request", http.StatusInternalServerError)
		return
	}

	client := &http.Client{Timeout: analyzeTimeout()}
	coreEndpoint := coreURL() + "/analyze"
	coreReq, err := http.NewRequest(http.MethodPost, coreEndpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		http.Error(w, "failed to build request to core service", http.StatusInternalServerError)
		return
	}
	coreReq.Header.Set("Content-Type", "application/json")
	coreReq.Header.Set("X-Request-ID", reqID)

	resp, err := client.Do(coreReq)
	if err != nil {
		log.Printf("[analyze][%s] failed to call core after=%s err=%v", reqID, time.Since(startedAt).Round(time.Millisecond), err)
		http.Error(w, "core service is unavailable", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	log.Printf(
		"[analyze][%s] core responded status=%d after=%s",
		reqID,
		resp.StatusCode,
		time.Since(startedAt).Round(time.Millisecond),
	)

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Request-ID", reqID)
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
	log.Printf("[analyze][%s] completed in=%s", reqID, time.Since(startedAt).Round(time.Millisecond))
}

func listDocumentsHandler(store *documentStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(store.list())
	}
}

func uploadDocumentHandler(store *documentStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if err := r.ParseMultipartForm(32 << 20); err != nil {
			http.Error(w, "invalid multipart form", http.StatusBadRequest)
			return
		}

		file, header, err := r.FormFile("file")
		if err != nil {
			http.Error(w, "missing file", http.StatusBadRequest)
			return
		}
		defer file.Close()

		ext := strings.ToLower(filepath.Ext(header.Filename))
		if ext != ".pdf" && ext != ".docx" {
			http.Error(w, "only .pdf and .docx are supported", http.StatusBadRequest)
			return
		}

		name := strings.TrimSpace(r.FormValue("name"))
		if name == "" {
			name = header.Filename
		}

		text := r.FormValue("text")
		item, err := store.create(name, text, file, header)
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to save document: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(item)
	}
}

func documentByIDHandler(store *documentStore) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/documents/")
		parts := strings.Split(strings.Trim(path, "/"), "/")
		if len(parts) == 0 || parts[0] == "" {
			http.NotFound(w, r)
			return
		}

		id := parts[0]
		if len(parts) == 2 && parts[1] == "download" {
			downloadDocument(store, w, r, id)
			return
		}

		switch r.Method {
		case http.MethodGet:
			item, ok := store.get(id)
			if !ok {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(item)
		case http.MethodPut:
			defer r.Body.Close()
			var req struct {
				Text string `json:"text"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, "invalid JSON body", http.StatusBadRequest)
				return
			}
			item, err := store.update(id, req.Text)
			if errors.Is(err, os.ErrNotExist) {
				http.NotFound(w, r)
				return
			}
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(item)
		case http.MethodDelete:
			err := store.remove(id)
			if errors.Is(err, os.ErrNotExist) {
				http.NotFound(w, r)
				return
			}
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusNoContent)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func downloadDocument(store *documentStore, w http.ResponseWriter, r *http.Request, id string) {
	item, ok := store.get(id)
	if !ok {
		http.NotFound(w, r)
		return
	}

	data, err := os.ReadFile(item.OriginalPath)
	if err != nil {
		http.Error(w, "file is unavailable", http.StatusInternalServerError)
		return
	}

	name := item.Name
	if !strings.HasSuffix(strings.ToLower(name), item.OriginalExt) {
		name += item.OriginalExt
	}

	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", name))
	w.Header().Set("Content-Type", "application/octet-stream")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

func waitForCoreService() {
	client := &http.Client{Timeout: 5 * time.Second}
	coreEndpoint := coreURL() + "/health"

	for {
		req, err := http.NewRequest(http.MethodGet, coreEndpoint, nil)
		if err != nil {
			log.Printf("failed to build health check request: %v", err)
			time.Sleep(2 * time.Second)
			continue
		}

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
	store, err := newDocumentStore("/app/storage/documents")
	if err != nil {
		log.Fatalf("failed to init document store: %v", err)
	}

	http.HandleFunc("/analyze", analyzeHandler)
	http.HandleFunc("/documents", listDocumentsHandler(store))
	http.HandleFunc("/documents/upload", uploadDocumentHandler(store))
	http.HandleFunc("/documents/", documentByIDHandler(store))

	addr := ":8080"
	log.Printf("Backend server is listening on %s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

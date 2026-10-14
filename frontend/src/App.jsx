import React, { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_DEFAULT =
  "https://research-intelligence-rag-production.up.railway.app";

const SUGGESTED_QUESTIONS = [
  "Summarize the main findings of the research papers.",
  "What methodologies are used in these papers?",
  "Compare the key findings and limitations.",
  "What are the major research gaps?",
];

const PIPELINE_STEPS = [
  {
    number: "01",
    title: "PDF ingestion",
    description:
      "PyMuPDF text extraction with Tesseract OCR fallback for scanned documents.",
  },
  {
    number: "02",
    title: "Chunking and embeddings",
    description:
      "Split extracted text into chunks and generate BGE embeddings.",
  },
  {
    number: "03",
    title: "Hybrid retrieval",
    description:
      "Combine dense vector retrieval with BM25 and rank relevant passages.",
  },
  {
    number: "04",
    title: "Grounded generation",
    description:
      "Use Gemini to generate an answer from retrieved evidence.",
  },
];

const TECHNOLOGIES = [
  {
    label: "API",
    title: "FastAPI",
    description: "Query and health endpoints",
  },
  {
    label: "LLM",
    title: "Google Gemini",
    description: "Answer generation",
  },
  {
    label: "Vectors",
    title: "Qdrant Cloud",
    description: "Vector storage and retrieval",
  },
  {
    label: "Search",
    title: "Dense + BM25",
    description: "Hybrid candidate retrieval",
  },
  {
    label: "Ranking",
    title: "Reranking",
    description: "Order evidence by relevance",
  },
  {
    label: "Storage",
    title: "PostgreSQL",
    description: "Application metadata and persistence",
  },
  {
    label: "Worker",
    title: "Ingestion worker",
    description: "Separate document processing service",
  },
];

const NAV_ITEMS = [
  { id: "overview", label: "Overview", icon: "◫" },
  { id: "research", label: "Research workspace", icon: "⌕" },
  { id: "library", label: "Document library", icon: "▤" },
  { id: "ingestion", label: "Ingestion pipeline", icon: "⇧" },
  { id: "settings", label: "Settings", icon: "⚙" },
];

function getInitials(text = "RI") {
  return text
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((word) => word[0].toUpperCase())
    .join("");
}

function formatDate(date) {
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function extractErrorMessage(data, fallback) {
  if (typeof data?.detail === "string") return data.detail;
  if (typeof data?.message === "string") return data.message;
  if (typeof data?.error === "string") return data.error;
  return fallback;
}

function normalizeSource(source, index) {
  return {
    id: source.chunk_id ?? `${source.document_id ?? "source"}-${index}`,
    documentId: source.document_id ?? "",
    documentName: source.document_name ?? "Research document",
    pageNumber: source.page_number ?? null,
    score: source.rerank_score ?? null,
    text: source.text ?? "",
  };
}

function formatScore(score) {
  if (score === null || score === undefined || score === "") {
    return "Not provided";
  }

  const numericScore = Number(score);

  return Number.isFinite(numericScore)
    ? numericScore.toFixed(4)
    : String(score);
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}


function renderCitedText(text, sources, onSelect) {
  const citationPattern = /\[(\d+)\]/g;
  const pieces = [];
  let lastIndex = 0;
  let match;
  let key = 0;

  while ((match = citationPattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      pieces.push(
        <ReactMarkdown
          key={`markdown-${key++}`}
          remarkPlugins={[remarkGfm]}
          components={{
            p: ({ children }) => <p>{children}</p>,
          }}
        >
          {text.slice(lastIndex, match.index)}
        </ReactMarkdown>
      );
    }

    const sourceNumber = Number(match[1]);
    const source = sources[sourceNumber - 1];

    if (source) {
      pieces.push(
        <button
          type="button"
          className="citation-link"
          key={`citation-${key++}`}
          title={`Open source ${sourceNumber}`}
          onClick={() => onSelect(source)}
        >
          [{sourceNumber}]
        </button>
      );
    } else {
      pieces.push(
        <span key={`unknown-${key++}`}>{match[0]}</span>
      );
    }

    lastIndex = citationPattern.lastIndex;
  }

  if (lastIndex < text.length) {
    pieces.push(
      <ReactMarkdown
        key={`markdown-${key++}`}
        remarkPlugins={[remarkGfm]}
      >
        {text.slice(lastIndex)}
      </ReactMarkdown>
    );
  }

  return pieces;
}


export default function App() {
  const [page, setPage] = useState("overview");
  const [apiUrl, setApiUrl] = useState(API_DEFAULT);
  const [liveMode, setLiveMode] = useState(true);
  const [darkMode, setDarkMode] = useState(true);

  const [apiStatus, setApiStatus] = useState("unknown");
  const [apiStatusMessage, setApiStatusMessage] = useState(
    "API connection has not been checked."
  );
  const [checkingHealth, setCheckingHealth] = useState(false);

  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [asking, setAsking] = useState(false);
  const [selectedSource, setSelectedSource] = useState(null);
  const [requestError, setRequestError] = useState("");

  const [topK, setTopK] = useState(5);
  const [copiedMessage, setCopiedMessage] = useState(false);

  const [libraryDocuments, setLibraryDocuments] = useState([]);
  const [uploadingDocument, setUploadingDocument] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");

  const chatEndRef = useRef(null);
  const questionInputRef = useRef(null);
  const documentFileInputRef = useRef(null);

  const cleanedApiUrl = useMemo(
    () => apiUrl.trim().replace(/\/+$/, ""),
    [apiUrl]
  );

  const totalSources = useMemo(
    () =>
      messages.reduce(
        (total, message) => total + (message.sources?.length ?? 0),
        0
      ),
    [messages]
  );

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  useEffect(() => {
    if (page === "research") {
      questionInputRef.current?.focus();
    }
  }, [page]);

  async function checkHealth() {
    if (!cleanedApiUrl) {
      setApiStatus("error");
      setApiStatusMessage("Enter a valid API base URL.");
      return;
    }

    setCheckingHealth(true);
    setApiStatus("checking");
    setApiStatusMessage("Checking the API health endpoint...");

    try {
      const response = await fetch(`${cleanedApiUrl}/health`, {
        method: "GET",
        headers: { Accept: "application/json" },
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          extractErrorMessage(
            data,
            `Health check failed with HTTP ${response.status}.`
          )
        );
      }

      setApiStatus("connected");
      setApiStatusMessage("The API health endpoint responded successfully.");
    } catch (error) {
      setApiStatus("error");
      setApiStatusMessage(
        error.message ||
          "Could not connect. Check the URL, network, and CORS configuration."
      );
    } finally {
      setCheckingHealth(false);
    }
  }

  async function askQuestion(questionText = question) {
    const query = questionText.trim();

    if (!query || asking) return;

    setQuestion("");
    setRequestError("");
    setSelectedSource(null);

    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: query,
      createdAt: new Date(),
    };

    setMessages((current) => [...current, userMessage]);
    setAsking(true);

    if (!liveMode) {
      setRequestError(
        "Live API mode is disabled. Enable it in Settings to submit this question to your deployed backend."
      );
      setAsking(false);
      return;
    }

    try {
      const response = await fetch(`${cleanedApiUrl}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ query }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          extractErrorMessage(
            data,
            `Query failed with HTTP ${response.status}.`
          )
        );
      }

      if (typeof data.answer !== "string") {
        throw new Error(
          "The API response did not contain an answer string."
        );
      }

      const rawSources = Array.isArray(data.sources) ? data.sources : [];
      const normalizedSources = rawSources.map(normalizeSource);

      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: data.answer,
        sources: normalizedSources,
        createdAt: new Date(),
      };

      setMessages((current) => [...current, assistantMessage]);
    } catch (error) {
      setRequestError(
        error.message ||
          "The request failed. Check the API URL and backend logs."
      );
    } finally {
      setAsking(false);
    }
  }

  async function uploadDocument(event) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";

    if (files.length === 0) return;

    const invalidFiles = files.filter(
      (file) => !file.name.toLowerCase().endsWith(".pdf")
    );

    if (invalidFiles.length > 0) {
      setUploadError(
        `Only PDF files are allowed. Invalid files: ${invalidFiles
          .map((file) => file.name)
          .join(", ")}`
      );
      setUploadMessage("");
      return;
    }

    if (!liveMode) {
      setUploadError(
        "Enable Live API mode in Settings before uploading documents."
      );
      setUploadMessage("");
      return;
    }

    setUploadingDocument(true);
    setUploadError("");
    setUploadMessage(`Uploading ${files.length} PDF(s)...`);

    const successfulUploads = [];
    const failedUploads = [];

    try {
      for (const file of files) {
        try {
          const formData = new FormData();
          formData.append("file", file);

          const response = await fetch(`${cleanedApiUrl}/documents/upload`, {
            method: "POST",
            headers: { Accept: "application/json" },
            body: formData,
          });

          const data = await response.json().catch(() => ({}));

          if (!response.ok) {
            throw new Error(
              extractErrorMessage(
                data,
                `Upload failed with HTTP ${response.status}.`
              )
            );
          }

          if (!data.document_id || !data.document_name) {
            throw new Error(
              "The upload endpoint returned an unexpected document response."
            );
          }

          successfulUploads.push({
            documentId: data.document_id,
            documentName: data.document_name,
            fileType: data.file_type ?? "PDF",
            pageCount: data.page_count ?? null,
            status: data.status ?? "registered",
            uploadedAt: new Date(),
          });
        } catch (error) {
          failedUploads.push({
            fileName: file.name,
            message: error.message || "Upload failed.",
          });
        }
      }

      if (successfulUploads.length > 0) {
        setLibraryDocuments((current) => {
          const updatedDocuments = [...current];

          for (const uploadedDocument of successfulUploads) {
            const existingIndex = updatedDocuments.findIndex(
              (existing) =>
                existing.documentId === uploadedDocument.documentId
            );

            if (existingIndex >= 0) {
              updatedDocuments.splice(existingIndex, 1);
            }

            updatedDocuments.unshift(uploadedDocument);
          }

          return updatedDocuments;
        });

        setApiStatusMessage(
          "Document upload endpoint responded successfully."
        );
      }

      if (failedUploads.length === 0) {
        setUploadMessage(
          `Successfully registered all ${successfulUploads.length} PDF(s).`
        );
        setUploadError("");
      } else {
        setUploadMessage(
          `Successfully registered ${successfulUploads.length} of ${files.length} PDF(s).`
        );

        setUploadError(
          failedUploads
            .map((failure) => `${failure.fileName}: ${failure.message}`)
            .join("\n")
        );
      }
    } finally {
      setUploadingDocument(false);
    }
  }

  function startNewChat() {
    setMessages([]);
    setQuestion("");
    setRequestError("");
    setSelectedSource(null);
    setCopiedMessage(false);
    setPage("research");
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessage(true);
      window.setTimeout(() => setCopiedMessage(false), 1600);
    } catch {
      setCopiedMessage(false);
      setRequestError(
        "Clipboard access was blocked by the browser. Select and copy the text manually."
      );
    }
  }

  function exportConversation() {
    const content = messages
      .map((message) => {
        const heading =
          message.role === "user" ? "QUESTION" : "ANSWER";

        const sources =
          message.sources?.length > 0
            ? "\n\nSources:\n" +
              message.sources
                .map(
                  (source, index) =>
                    `[${index + 1}] ${source.documentName}${
                      source.pageNumber != null
                        ? ` — page ${source.pageNumber}`
                        : ""
                    }\n${source.text}`
                )
                .join("\n\n")
            : "";

        return `${heading}\n${message.content}${sources}`;
      })
      .join("\n\n----------------------------------------\n\n");

    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");

    anchor.href = url;
    anchor.download = "research-conversation.txt";
    anchor.click();

    URL.revokeObjectURL(url);
  }

  return (
    <div className={`app-shell ${darkMode ? "theme-dark" : "theme-light"}`}>
      <div className="app-layout">
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-mark">R</div>
            <div>
              <p className="brand-name">Research Intelligence</p>
              <p className="brand-caption">
                Evidence-first research workspace
              </p>
            </div>
          </div>

          <div>
            <p className="nav-label">Workspace</p>
            <nav className="nav-list" aria-label="Main navigation">
              {NAV_ITEMS.map((item) => (
                <button
                  type="button"
                  key={item.id}
                  className={`nav-item ${page === item.id ? "active" : ""}`}
                  onClick={() => setPage(item.id)}
                >
                  <span className="nav-icon">{item.icon}</span>
                  <span>{item.label}</span>
                </button>
              ))}
            </nav>
          </div>

          <div className="sidebar-spacer" />

          <div className="sidebar-status">
            <div className="status-heading">
              <span
                className={`status-dot ${
                  apiStatus === "connected"
                    ? "connected"
                    : apiStatus === "error"
                    ? "error"
                    : ""
                }`}
              />
              <span>
                {apiStatus === "connected"
                  ? "API reachable"
                  : apiStatus === "checking"
                  ? "Checking API"
                  : apiStatus === "error"
                  ? "API unavailable"
                  : "API not checked"}
              </span>
            </div>

            <p className="status-description">{apiStatusMessage}</p>

            <button
              type="button"
              className="button button-small"
              style={{ width: "100%", marginTop: 13 }}
              onClick={checkHealth}
              disabled={checkingHealth}
            >
              {checkingHealth ? "Checking..." : "Check connection"}
            </button>
          </div>

          <div className="sidebar-footer">Research Intelligence RAG</div>
        </aside>

        <main className="main-area">
          <header className="topbar">
            <div>
              <p className="breadcrumb">Research Intelligence / Workspace</p>
              <h1 className="topbar-title">
                {NAV_ITEMS.find((item) => item.id === page)?.label ??
                  "Workspace"}
              </h1>
            </div>

            <div className="topbar-actions">
              <button
                type="button"
                className="button button-quiet"
                onClick={() => setPage("settings")}
              >
                <span>⚙</span>
                <span>Settings</span>
              </button>

              <button
                type="button"
                className="button button-primary"
                onClick={startNewChat}
              >
                <span>＋</span>
                <span>New research</span>
              </button>
            </div>
          </header>

          <div className="content">
            {page === "overview" && (
              <>
                <section className="page-heading">
                  <div>
                    <p className="eyebrow">Research intelligence platform</p>
                    <h2 className="page-title">
                      Turn research papers into insights.
                    </h2>
                    <p className="page-description">
                      Explore a retrieval-augmented generation pipeline
                      designed to retrieve relevant evidence from research
                      documents and generate grounded answers.
                    </p>
                  </div>

                  <button
                    type="button"
                    className="button button-primary"
                    onClick={() => setPage("research")}
                  >
                    Open research workspace →
                  </button>
                </section>

                <section
                  className="grid grid-three"
                  style={{ marginBottom: 21 }}
                >
                  <div className="metric-card">
                    <div className="metric-top">
                      <span className="metric-label">API status</span>
                      <span className="metric-icon">⌁</span>
                    </div>
                    <p className="metric-value" style={{ fontSize: 21 }}>
                      {apiStatus === "connected"
                        ? "Reachable"
                        : apiStatus === "checking"
                        ? "Checking..."
                        : apiStatus === "error"
                        ? "Unavailable"
                        : "Unchecked"}
                    </p>
                    <p className="metric-note">
                      Based on the latest health check, not a complete system
                      diagnostics report.
                    </p>
                  </div>

                  <div className="metric-card">
                    <div className="metric-top">
                      <span className="metric-label">Questions this session</span>
                      <span className="metric-icon">⌕</span>
                    </div>
                    <p className="metric-value">
                      {messages.filter((message) => message.role === "user").length}
                    </p>
                    <p className="metric-note">
                      Questions in the current browser session.
                    </p>
                  </div>

                  <div className="metric-card">
                    <div className="metric-top">
                      <span className="metric-label">Retrieved sources</span>
                      <span className="metric-icon">▤</span>
                    </div>
                    <p className="metric-value">{totalSources}</p>
                    <p className="metric-note">
                      Sources returned by successful query requests in this
                      session.
                    </p>
                  </div>
                </section>

                <section className="grid grid-two">
                  <div className="panel">
                    <div className="panel-header">
                      <div>
                        <h3 className="panel-title">
                          Retrieval and generation pipeline
                        </h3>
                        <p className="panel-subtitle">Project architecture</p>
                      </div>
                    </div>

                    <div className="panel-body">
                      <div className="pipeline-list">
                        {PIPELINE_STEPS.map((step) => (
                          <div className="pipeline-step" key={step.number}>
                            <div className="step-number">{step.number}</div>
                            <div>
                              <h4 className="step-title">{step.title}</h4>
                              <p className="step-description">
                                {step.description}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="panel">
                    <div className="panel-header">
                      <div>
                        <h3 className="panel-title">Technology stack</h3>
                        <p className="panel-subtitle">
                          Components used by the project
                        </p>
                      </div>
                    </div>

                    <div className="panel-body">
                      <div className="technology-list">
                        {TECHNOLOGIES.map((technology) => (
                          <div
                            className="technology-row"
                            key={technology.title}
                          >
                            <div className="technology-icon">
                              {technology.label}
                            </div>
                            <div className="technology-content">
                              <h4 className="technology-title">
                                {technology.title}
                              </h4>
                              <p className="technology-description">
                                {technology.description}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </section>
              </>
            )}
            {page === "research" && (
              <>
                <section className="page-heading">
                  <div>
                    <p className="eyebrow">Evidence-grounded Q&A</p>
                    <h2 className="page-title">Research workspace</h2>
                    <p className="page-description">
                      Ask questions about indexed research documents. Answers
                      and source passages depend on the configured backend.
                    </p>
                  </div>
                  <button
                    type="button"
                    className="button"
                    onClick={startNewChat}
                  >
                    New conversation
                  </button>
                </section>

                <div className="workspace">
                  <section className="panel chat-panel">
                    <div className="panel-header">
                      <div>
                        <h3 className="panel-title">Research conversation</h3>
                        <p className="panel-subtitle">
                          Answers are generated from retrieved evidence.
                        </p>
                      </div>
                      <div className="chat-header-actions">
                        <button
                          type="button"
                          className="button button-small"
                          disabled={messages.length === 0}
                          onClick={exportConversation}
                        >
                          Export
                        </button>
                        <button
                          type="button"
                          className="button button-small"
                          onClick={startNewChat}
                        >
                          Clear
                        </button>
                      </div>
                    </div>

                    <div className="chat-messages">
                      {messages.length === 0 && (
                        <div className="empty-chat">
                          <div className="empty-icon">⌕</div>
                          <h3 className="empty-title">
                            What would you like to research?
                          </h3>
                          <p className="empty-description">
                            Ask a question about your indexed research
                            documents. Enable Live API mode in Settings to
                            submit requests to your backend.
                          </p>
                          <div className="suggestions">
                            {SUGGESTED_QUESTIONS.map((suggestion) => (
                              <button
                                type="button"
                                className="suggestion"
                                key={suggestion}
                                onClick={() => {
                                  setQuestion(suggestion);
                                  questionInputRef.current?.focus();
                                }}
                              >
                                {suggestion}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      {messages.map((message) => (
                        <article className="message" key={message.id}>
                          <div
                            className={`message-avatar ${
                              message.role === "user" ? "user-avatar" : ""
                            }`}
                          >
                            {message.role === "user" ? "You" : "RI"}
                          </div>
                          <div className="message-content">
                            <div className="message-heading">
                              <span className="message-author">
                                {message.role === "user"
                                  ? "You"
                                  : "Research Intelligence"}
                              </span>
                              <span className="message-time">
                                {formatDate(message.createdAt)}
                              </span>
                            </div>
                            <div className="message-text">
                              {message.role === "assistant"
                                ? renderCitedText(
                                    message.content,
                                    message.sources ?? [],
                                    setSelectedSource
                                  )
                                : message.content}
                            </div>

                            {message.role === "assistant" && (
                              <div className="message-actions">
                                <button
                                  type="button"
                                  className="button button-small"
                                  onClick={() => copyText(message.content)}
                                >
                                  {copiedMessage ? "Copied" : "Copy answer"}
                                </button>
                              </div>
                            )}
                          </div>
                        </article>
                      ))}

                      {asking && (
                        <div className="message">
                          <div className="message-avatar">RI</div>
                          <div className="message-content">
                            <div className="message-heading">
                              <span className="message-author">
                                Research Intelligence
                              </span>
                            </div>
                            <p className="message-text">
                              Retrieving evidence and preparing an answer...
                            </p>
                          </div>
                        </div>
                      )}

                      <div ref={chatEndRef} />
                    </div>

                    <div className="chat-composer">
                      {requestError && (
                        <div className="notice error" role="alert">
                          {requestError}
                        </div>
                      )}

                      <form
                        onSubmit={(event) => {
                          event.preventDefault();
                          askQuestion();
                        }}
                      >
                        <div className="composer-box">
                          <textarea
                            ref={questionInputRef}
                            className="composer-input"
                            value={question}
                            onChange={(event) =>
                              setQuestion(event.target.value)
                            }
                            placeholder="Ask a question about your research..."
                            rows={2}
                            aria-label="Research question"
                          />
                          <button
                            type="submit"
                            className="button button-primary"
                            disabled={asking || !question.trim()}
                          >
                            {asking ? "Working..." : "Ask"}
                          </button>
                        </div>
                      </form>

                      <p className="composer-footnote">
                        Verify generated conclusions against the cited source
                        passages and original papers.
                      </p>
                    </div>
                  </section>

                  <aside className="workspace-side">
                    <section className="panel side-panel">
                      <h3 className="side-heading">Retrieved sources</h3>
                      <p className="side-description">
                        Select a source to inspect the passage returned by the
                        backend.
                      </p>

                      {messages.some(
                        (message) =>
                          message.role === "assistant" &&
                          message.sources?.length > 0
                      ) ? (
                        <div className="source-list">
                          {messages
                            .filter(
                              (message) =>
                                message.role === "assistant" &&
                                message.sources?.length > 0
                            )
                            .flatMap((message) => message.sources)
                            .map((source, index) => (
                              <button
                                type="button"
                                className={`source-card ${
                                  selectedSource?.id === source.id
                                    ? "selected"
                                    : ""
                                }`}
                                key={`${source.id}-${index}`}
                                onClick={() => setSelectedSource(source)}
                              >
                                <span className="source-number">
                                  {index + 1}
                                </span>
                                <span className="source-name">
                                  {source.documentName}
                                </span>
                                <p className="source-meta">
                                  {source.pageNumber != null
                                    ? `Page ${source.pageNumber}`
                                    : "Page not provided"}
                                  {" · "}
                                  Score: {formatScore(source.score)}
                                </p>
                              </button>
                            ))}
                        </div>
                      ) : (
                        <p className="empty-small">
                          No sources yet. Submit a question in Live API mode to
                          see retrieved evidence here.
                        </p>
                      )}

                      {selectedSource && (
                        <div className="source-preview">
                          <h4 className="step-title">
                            {selectedSource.documentName}
                          </h4>
                          <p className="source-meta">
                            {selectedSource.pageNumber != null
                              ? `Page ${selectedSource.pageNumber}`
                              : "Page not provided"}
                          </p>
                          <p className="source-preview-text">
                            {selectedSource.text ||
                              "The backend did not return passage text."}
                          </p>
                        </div>
                      )}
                    </section>

                    <section className="panel side-panel">
                      <h3 className="side-heading">Query settings</h3>
                      <p className="side-description">
                        Retrieval depth is controlled by the backend unless
                        its request schema explicitly supports a parameter.
                      </p>
                      <div className="callout">
                        <span className="callout-icon">ⓘ</span>
                        <div>
                          <p className="callout-title">Backend controls</p>
                          <p className="callout-text">
                            The UI does not send unsupported retrieval
                            parameters. Configure retrieval settings in the
                            backend when those controls are available.
                          </p>
                        </div>
                      </div>
                    </section>
                  </aside>
                </div>
              </>
            )}

            {page === "library" && (
              <>
                <section className="page-heading">
                  <div>
                    <p className="eyebrow">Research document collection</p>
                    <h2 className="page-title">Document library</h2>
                    <p className="page-description">
                      Upload PDFs through the backend document-upload
                      endpoint. This list shows documents uploaded during the
                      current frontend session.
                    </p>
                  </div>
                </section>

                <section className="panel" style={{ marginBottom: 20 }}>
                  <div className="panel-header">
                    <div>
                      <h3 className="panel-title">Upload a research paper</h3>
                      <p className="panel-subtitle">
                        POST /documents/upload · PDF files
                      </p>
                    </div>
                    <span
                      className={`status-pill ${
                        uploadingDocument
                          ? ""
                          : uploadMessage
                          ? "success"
                          : ""
                      }`}
                    >
                      {uploadingDocument
                        ? "Uploading"
                        : uploadMessage
                        ? "Last upload complete"
                        : "Ready"}
                    </span>
                  </div>

                  <div className="panel-body">
                    <input
                      ref={documentFileInputRef}
                      type="file"
                      accept="application/pdf,.pdf"
                      multiple
                      onChange={uploadDocument}
                      style={{ display: "none" }}
                      aria-label="Choose multiple PDF documents"
                    />

                    <div className="callout">
                      <span className="callout-icon">ⓘ</span>
                      <div>
                        <p className="callout-title">Backend registration</p>
                        <p className="callout-text">
                          The backend accepts the PDF and returns its document
                          ID, name, file type, page count, and status. Ingestion
                          may be handled asynchronously by the separate worker.
                          A successful upload does not necessarily mean the
                          document is fully indexed.
                        </p>
                      </div>
                    </div>

                    <button
                      type="button"
                      className="button button-primary"
                      style={{ marginTop: 18 }}
                      onClick={() => documentFileInputRef.current?.click()}
                      disabled={uploadingDocument}
                    >
                      {uploadingDocument
                        ? "Uploading PDFs..."
                        : "Choose PDFs and upload"}
                    </button>

                    {!liveMode && (
                      <p className="field-help" style={{ marginTop: 12 }}>
                        Live API mode is disabled. Enable it in Settings to
                        upload through the configured API.
                      </p>
                    )}

                    {uploadError && (
                      <div
                        className="notice error"
                        role="alert"
                        style={{ marginTop: 16, marginBottom: 0 }}
                      >
                        {uploadError}
                      </div>
                    )}

                    {uploadMessage && (
                      <div
                        className="notice success"
                        role="status"
                        style={{ marginTop: 16, marginBottom: 0 }}
                      >
                        {uploadMessage}
                      </div>
                    )}
                  </div>
                </section>

                <section className="panel">
                  <div className="panel-header">
                    <div>
                      <h3 className="panel-title">
                        Documents uploaded this session
                      </h3>
                      <p className="panel-subtitle">
                        {libraryDocuments.length} document
                        {libraryDocuments.length === 1 ? "" : "s"} registered
                        in this browser session
                      </p>
                    </div>
                  </div>

                  {libraryDocuments.length > 0 ? (
                    <div className="table-wrap">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Document</th>
                            <th>Type</th>
                            <th>Pages</th>
                            <th>Status</th>
                            <th>Uploaded</th>
                          </tr>
                        </thead>
                        <tbody>
                          {libraryDocuments.map((document) => (
                            <tr key={document.documentId}>
                              <td>
                                <strong>{document.documentName}</strong>
                                <p className="source-meta">
                                  ID: {document.documentId}
                                </p>
                              </td>
                              <td>{document.fileType}</td>
                              <td>{document.pageCount ?? "Not provided"}</td>
                              <td>
                                <span className="status-pill">
                                  {document.status}
                                </span>
                              </td>
                              <td>{formatDate(document.uploadedAt)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="library-placeholder">
                      <div className="placeholder-icon">▤</div>
                      <h3 className="placeholder-title">
                        No documents uploaded in this session
                      </h3>
                      <p className="placeholder-text">
                        Select a PDF above to register it with the backend.
                        This list is session-local and does not fetch a
                        document inventory from the server.
                      </p>
                    </div>
                  )}
                </section>
              </>
            )}
            {page === "ingestion" && (
              <>
                <section className="page-heading">
                  <div>
                    <p className="eyebrow">Document processing</p>
                    <h2 className="page-title">Ingestion pipeline</h2>
                    <p className="page-description">
                      An overview of the research document ingestion and
                      retrieval stages used by this project.
                    </p>
                  </div>
                </section>

                <section className="grid grid-two">
                  <div className="panel">
                    <div className="panel-header">
                      <div>
                        <h3 className="panel-title">Pipeline stages</h3>
                        <p className="panel-subtitle">
                          From source PDF to grounded response
                        </p>
                      </div>
                    </div>
                    <div className="panel-body">
                      <div className="pipeline-list">
                        {PIPELINE_STEPS.map((step) => (
                          <div className="pipeline-step" key={step.number}>
                            <div className="step-number">{step.number}</div>
                            <div>
                              <h4 className="step-title">{step.title}</h4>
                              <p className="step-description">
                                {step.description}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="panel">
                    <div className="panel-header">
                      <div>
                        <h3 className="panel-title">Processing boundaries</h3>
                        <p className="panel-subtitle">
                          What this frontend can verify
                        </p>
                      </div>
                    </div>
                    <div className="panel-body">
                      <div className="callout">
                        <span className="callout-icon">ⓘ</span>
                        <div>
                          <p className="callout-title">
                            Asynchronous ingestion
                          </p>
                          <p className="callout-text">
                            The separate ingestion worker may process uploaded
                            documents asynchronously. This page does not
                            invent a job-status endpoint or claim that
                            ingestion has completed without a confirmed
                            backend response.
                          </p>
                        </div>
                      </div>
                      <button
                        type="button"
                        className="button button-primary"
                        style={{ marginTop: 16 }}
                        onClick={() => setPage("library")}
                      >
                        Review document uploads
                      </button>
                    </div>
                  </div>
                </section>
              </>
            )}

            {page === "settings" && (
              <>
                <section className="page-heading">
                  <div>
                    <p className="eyebrow">Workspace configuration</p>
                    <h2 className="page-title">Settings</h2>
                    <p className="page-description">
                      Configure the API connection and frontend preferences.
                    </p>
                  </div>
                </section>

                <section className="grid grid-two">
                  <div className="panel">
                    <div className="panel-header">
                      <div>
                        <h3 className="panel-title">Backend connection</h3>
                        <p className="panel-subtitle">
                          FastAPI deployment configuration
                        </p>
                      </div>
                      <span
                        className={`status-pill ${
                          apiStatus === "connected" ? "success" : ""
                        }`}
                      >
                        {apiStatus === "connected"
                          ? "Reachable"
                          : apiStatus === "checking"
                          ? "Checking"
                          : apiStatus === "error"
                          ? "Error"
                          : "Unchecked"}
                      </span>
                    </div>

                    <div className="panel-body">
                      <div className="field">
                        <label className="field-label" htmlFor="api-url">
                          API base URL
                        </label>
                        <input
                          id="api-url"
                          className="text-input"
                          value={apiUrl}
                          onChange={(event) => {
                            setApiUrl(event.target.value);
                            setApiStatus("unknown");
                            setApiStatusMessage(
                              "API URL changed. Run a new health check."
                            );
                          }}
                          placeholder="https://your-api.example.com"
                        />
                        <p className="field-help">
                          Enter the base URL only. The frontend appends
                          /health, /query, and /documents/upload to it.
                        </p>
                      </div>

                      <div className="notice">
                        <strong>Current health status:</strong>{" "}
                        {apiStatusMessage}
                      </div>

                      <div
                        style={{
                          display: "flex",
                          flexWrap: "wrap",
                          gap: 9,
                        }}
                      >
                        <button
                          type="button"
                          className="button button-primary"
                          onClick={checkHealth}
                          disabled={checkingHealth}
                        >
                          {checkingHealth
                            ? "Checking connection..."
                            : "Test API connection"}
                        </button>

                        <button
                          type="button"
                          className="button"
                          onClick={() => {
                            setApiUrl(API_DEFAULT);
                            setApiStatus("unknown");
                            setApiStatusMessage(
                              "Default API URL restored. Run a health check."
                            );
                          }}
                        >
                          Reset URL
                        </button>
                      </div>

                      <div className="setting-row" style={{ marginTop: 16 }}>
                        <div>
                          <p className="setting-name">Live API mode</p>
                          <p className="setting-description">
                            Send research questions and PDF uploads to the
                            configured backend.
                          </p>
                        </div>
                        <button
                          type="button"
                          className={`toggle ${liveMode ? "on" : ""}`}
                          role="switch"
                          aria-checked={liveMode}
                          aria-label="Enable live API mode"
                          onClick={() => setLiveMode((current) => !current)}
                        >
                          <span className="toggle-knob" />
                        </button>
                      </div>

                      {liveMode && (
                        <p className="field-help" style={{ marginTop: 12 }}>
                          Live mode is enabled. A successful health check
                          verifies /health only. Queries and uploads are tested
                          when you submit them.
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="panel">
                    <div className="panel-header">
                      <div>
                        <h3 className="panel-title">Interface preferences</h3>
                        <p className="panel-subtitle">
                          Customize your workspace
                        </p>
                      </div>
                    </div>

                    <div className="panel-body">
                      <div className="setting-row">
                        <div>
                          <p className="setting-name">Dark theme</p>
                          <p className="setting-description">
                            Switch between dark and light appearance.
                          </p>
                        </div>
                        <button
                          type="button"
                          className={`toggle ${darkMode ? "on" : ""}`}
                          role="switch"
                          aria-checked={darkMode}
                          aria-label="Enable dark theme"
                          onClick={() => setDarkMode((current) => !current)}
                        >
                          <span className="toggle-knob" />
                        </button>
                      </div>

                      <div className="setting-row">
                        <div>
                          <p className="setting-name">Session questions</p>
                          <p className="setting-description">
                            Questions currently stored in frontend memory.
                          </p>
                        </div>
                        <span className="status-pill">
                          {
                            messages.filter(
                              (message) => message.role === "user"
                            ).length
                          }{" "}
                          questions
                        </span>
                      </div>

                      <div className="setting-row">
                        <div>
                          <p className="setting-name">Conversation history</p>
                          <p className="setting-description">
                            Clear the current in-memory conversation.
                          </p>
                        </div>
                        <button
                          type="button"
                          className="button button-small"
                          onClick={startNewChat}
                        >
                          Clear history
                        </button>
                      </div>

                      <div className="setting-row">
                        <div>
                          <p className="setting-name">Export conversation</p>
                          <p className="setting-description">
                            Download the current conversation as a text file.
                          </p>
                        </div>
                        <button
                          type="button"
                          className="button button-small"
                          disabled={messages.length === 0}
                          onClick={exportConversation}
                        >
                          Export
                        </button>
                      </div>
                    </div>
                  </div>
                </section>

                <div style={{ marginTop: 19 }}>
                  <div className="callout">
                    <span className="callout-icon">ⓘ</span>
                    <div>
                      <p className="callout-title">Configuration scope</p>
                      <p className="callout-text">
                        These preferences live in the current React session.
                        This version does not modify server environment
                        variables, database records, Qdrant collections, or
                        ingestion-worker settings.
                      </p>
                    </div>
                  </div>
                </div>
              </>
            )}

            <p className="footer-note">
              Research Intelligence RAG · Answers should be verified against
              retrieved evidence.
            </p>
          </div>
        </main>
      </div>
    </div>
  );
}
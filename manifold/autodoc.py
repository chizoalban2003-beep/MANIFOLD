"""Phase 55: Self-Documenting API — Living Documentation Engine.

Turns the MANIFOLD codebase into a living resource by:

1. **DocExtractor** — uses the :mod:`inspect` module to crawl all modules,
   reading docstrings and extracting class/function signatures.
2. **APIExplorer** — generates a pure-HTML/CSS interactive API explorer page
   (zero-dependency Swagger alternative) that lets users test endpoints
   directly from the browser.

Key classes
-----------
``DocEntry``
    Documentation record for one symbol (module, class, or function).
``ModuleDoc``
    All documentation records extracted from one module.
``DocExtractor``
    Crawls modules and returns :class:`ModuleDoc` objects.
``APIEndpoint``
    Descriptor for one HTTP endpoint.
``APIExplorer``
    Generates an interactive HTML/CSS API explorer page.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# DocEntry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocEntry:
    """Documentation record for one Python symbol.

    Attributes
    ----------
    name:
        Symbol name (e.g. ``"VectorIndex"`` or ``"cosine_similarity"``).
    kind:
        ``"class"`` | ``"function"`` | ``"method"`` | ``"module"``.
    docstring:
        The ``__doc__`` string, stripped of leading/trailing whitespace.
        Empty string if no docstring.
    signature:
        String representation of the call signature (for callables).
        Empty string for non-callables.
    module_name:
        Dotted module name where the symbol was found.
    """

    name: str
    kind: str
    docstring: str
    signature: str
    module_name: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "kind": self.kind,
            "docstring": self.docstring[:500] if self.docstring else "",
            "signature": self.signature,
            "module_name": self.module_name,
        }


# ---------------------------------------------------------------------------
# ModuleDoc
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModuleDoc:
    """All documentation records extracted from one module.

    Attributes
    ----------
    module_name:
        Dotted module name.
    module_docstring:
        The module's top-level ``__doc__`` string.
    entries:
        Tuple of :class:`DocEntry` for public symbols in this module.
    """

    module_name: str
    module_docstring: str
    entries: tuple[DocEntry, ...]

    @property
    def public_classes(self) -> list[DocEntry]:
        """Return entries whose kind is ``"class"``."""
        return [e for e in self.entries if e.kind == "class"]

    @property
    def public_functions(self) -> list[DocEntry]:
        """Return entries whose kind is ``"function"``."""
        return [e for e in self.entries if e.kind == "function"]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "module_name": self.module_name,
            "module_docstring": self.module_docstring[:300] if self.module_docstring else "",
            "class_count": len(self.public_classes),
            "function_count": len(self.public_functions),
            "entries": [e.to_dict() for e in self.entries],
        }


# ---------------------------------------------------------------------------
# DocExtractor
# ---------------------------------------------------------------------------


@dataclass
class DocExtractor:
    """Crawls Python modules and extracts structured documentation.

    Parameters
    ----------
    package_name:
        Top-level package to document (default: ``"manifold"``).
    include_private:
        If ``True``, include symbols whose names start with ``"_"``.
        Default: ``False``.
    max_docstring_length:
        Truncate docstrings to this many characters in output.
        Default: ``2000``.

    Example
    -------
    ::

        extractor = DocExtractor()
        module_docs = extractor.extract_module("manifold.vectorfs")
        for entry in module_docs.entries:
            print(f"{entry.kind}: {entry.name}")
    """

    package_name: str = "manifold"
    include_private: bool = False
    max_docstring_length: int = 2_000

    def extract_module(self, module_name: str) -> ModuleDoc:
        """Extract documentation from one module.

        Parameters
        ----------
        module_name:
            Dotted module path (e.g. ``"manifold.vectorfs"``).

        Returns
        -------
        ModuleDoc
            Documentation records for the module and its public symbols.
        """
        import importlib

        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            return ModuleDoc(
                module_name=module_name,
                module_docstring=f"[Import error: {exc}]",
                entries=(),
            )

        mod_doc = (mod.__doc__ or "").strip()
        entries: list[DocEntry] = []

        for name, obj in inspect.getmembers(mod):
            if not self.include_private and name.startswith("_"):
                continue
            # Only document items defined in this module
            obj_module = getattr(obj, "__module__", None)
            if obj_module != module_name:
                continue

            if inspect.isclass(obj):
                entries.extend(self._extract_class(name, obj, module_name))
            elif inspect.isfunction(obj):
                entries.append(self._extract_callable(name, obj, "function", module_name))

        return ModuleDoc(
            module_name=module_name,
            module_docstring=mod_doc[: self.max_docstring_length],
            entries=tuple(entries),
        )

    def extract_all(self, module_names: list[str]) -> list[ModuleDoc]:
        """Extract documentation from a list of module names.

        Parameters
        ----------
        module_names:
            List of dotted module paths to process.

        Returns
        -------
        list[ModuleDoc]
            One entry per module, in the order given.
        """
        return [self.extract_module(name) for name in module_names]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_callable(
        self,
        name: str,
        obj: Any,
        kind: str,
        module_name: str,
    ) -> DocEntry:
        """Build a :class:`DocEntry` for a callable."""
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            sig = "(...)"
        doc = (obj.__doc__ or "").strip()[: self.max_docstring_length]
        return DocEntry(
            name=name,
            kind=kind,
            docstring=doc,
            signature=f"{name}{sig}",
            module_name=module_name,
        )

    def _extract_class(
        self, name: str, cls: type, module_name: str
    ) -> list[DocEntry]:
        """Build DocEntry objects for a class and its public methods."""
        entries: list[DocEntry] = []
        doc = (cls.__doc__ or "").strip()[: self.max_docstring_length]
        try:
            sig = str(inspect.signature(cls))
        except (ValueError, TypeError):
            sig = "(...)"

        entries.append(
            DocEntry(
                name=name,
                kind="class",
                docstring=doc,
                signature=f"{name}{sig}",
                module_name=module_name,
            )
        )

        for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not self.include_private and method_name.startswith("_"):
                continue
            entries.append(
                self._extract_callable(
                    f"{name}.{method_name}", method, "method", module_name
                )
            )

        return entries


# ---------------------------------------------------------------------------
# APIEndpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class APIEndpoint:
    """Descriptor for one HTTP endpoint shown in the API explorer.

    Attributes
    ----------
    method:
        HTTP method (``"GET"`` or ``"POST"``).
    path:
        URL path (e.g. ``"/vector/add"``).
    summary:
        One-line description.
    description:
        Multi-sentence description shown in the explorer.
    request_example:
        JSON string example for the request body (``""`` for GET).
    response_example:
        JSON string example for the response body.
    tags:
        Category tags for grouping.
    """

    method: str
    path: str
    summary: str
    description: str = ""
    request_example: str = ""
    response_example: str = ""
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "method": self.method,
            "path": self.path,
            "summary": self.summary,
            "description": self.description,
            "request_example": self.request_example,
            "response_example": self.response_example,
            "tags": list(self.tags),
        }


# ---------------------------------------------------------------------------
# MANIFOLD standard endpoints
# ---------------------------------------------------------------------------

MANIFOLD_ENDPOINTS: list[APIEndpoint] = [
    APIEndpoint(
        method="POST", path="/shield",
        summary="Run a BrainTask through the ActiveInterceptor",
        description="Evaluates a task against the 4-vector grid and returns a veto decision.",
        request_example='{"prompt":"summarise report","domain":"general","stakes":0.5,"uncertainty":0.3,"complexity":0.4}',
        response_example='{"vetoed":false,"reason":"risk within threshold","risk_score":0.12,"confidence":0.87,"suggested_action":"use_tool"}',
        tags=("brain", "interceptor"),
    ),
    APIEndpoint(
        method="POST", path="/vector/add",
        summary="Add a vector embedding to the VectorIndex",
        description="Stores a high-dimensional embedding with optional metadata. Uses LSH for O(1) search.",
        request_example='{"vector_id":"doc-1","vector":[0.1,0.2,0.9],"metadata":{"source":"paper"}}',
        response_example='{"stored":true,"vector_id":"doc-1","bucket_key":"101"}',
        tags=("vectorfs", "semantic"),
    ),
    APIEndpoint(
        method="POST", path="/vector/search",
        summary="Cosine similarity nearest-neighbour search",
        description="Returns the top-k most similar vectors from the matching LSH bucket.",
        request_example='{"query":[0.1,0.2,0.9],"k":5}',
        response_example='{"results":[{"vector_id":"doc-1","similarity":0.999}]}',
        tags=("vectorfs", "semantic"),
    ),
    APIEndpoint(
        method="POST", path="/sandbox/execute",
        summary="Execute sandboxed Python code",
        description="Validates and executes code through the AST validator and budgeted executor.",
        request_example='{"code":"x = 1 + 1","agent_id":"agent-1","budget":1000}',
        response_example='{"success":true,"stdout":"","stderr":"","instructions_used":12}',
        tags=("sandbox", "security"),
    ),
    APIEndpoint(
        method="POST", path="/dag/execute",
        summary="Execute a multi-step DAG workflow",
        description="Runs a Directed Acyclic Graph of BrainTasks through topological sort.",
        request_example='{"graph_id":"pipeline-1","nodes":[{"node_id":"a","task":{"prompt":"step 1","domain":"general","stakes":0.3}},{"node_id":"b","task":{"prompt":"step 2","domain":"general","stakes":0.3},"depends_on":["a"]}]}',
        response_example='{"graph_id":"pipeline-1","all_succeeded":true,"total_nodes":2,"succeeded":2,"failed":0}',
        tags=("dag", "orchestration"),
    ),
    APIEndpoint(
        method="POST", path="/meta/outcome",
        summary="Record a task outcome for A/B prompt testing",
        description="Updates the PromptGenome success rate. Promotes Challenger to Champion if it outperforms by 5% over 100 trials.",
        request_example='{"success":true,"genome_id":"challenger-v1"}',
        response_example='{"recorded":true,"promotions":0}',
        tags=("meta", "evolution"),
    ),
    APIEndpoint(
        method="GET", path="/dashboard",
        summary="Fleet Dashboard UI",
        description="Returns the full HTML fleet dashboard with all phase panels.",
        response_example="<HTML dashboard>",
        tags=("ui", "monitoring"),
    ),
    APIEndpoint(
        method="GET", path="/admin/metrics",
        summary="JSON metrics snapshot",
        description="Returns real-time system metrics: PID threshold, entropy, peer counts, etc.",
        response_example='{"pid_threshold":0.45,"system_entropy":0.12,"vector_count":0}',
        tags=("admin", "monitoring"),
    ),
    APIEndpoint(
        method="POST", path="/b2b/handshake",
        summary="B2B policy handshake",
        description="Perform a cross-organisation trust handshake via B2BRouter.",
        request_example='{"org_id":"partner-org","data_classification":"internal","max_risk":0.4,"required_uptime":0.99,"allowed_tools":["gpt-4o"],"blocked_tools":[]}',
        response_example='{"compatible":true,"reason":"Policies compatible","trust_score":0.88}',
        tags=("b2b", "trust"),
    ),
    APIEndpoint(
        method="GET", path="/reputation/<id>",
        summary="Query reputation score for a tool",
        description="Returns the current reliability score for a tool or agent from the ReputationHub.",
        response_example='{"tool_name":"gpt-4o","reliability":0.92,"is_probationary":false}',
        tags=("reputation", "trust"),
    ),
    APIEndpoint(
        method="POST", path="/multisig/endorse",
        summary="Submit a multi-sig peer endorsement",
        description="High-stakes actions (asset > 0.9) require M-of-N peer signatures before execution.",
        request_example='{"proposal_id":"prop-1","signer_id":"peer-a","signature":"abc123"}',
        response_example='{"endorsed":true,"signature_count":2,"threshold":2}',
        tags=("multisig", "security"),
    ),
    APIEndpoint(
        method="GET", path="/docs",
        summary="Interactive API Explorer",
        description="This page — zero-dependency Swagger-style HTML explorer.",
        response_example="<HTML docs>",
        tags=("docs",),
    ),
]


# ---------------------------------------------------------------------------
# APIExplorer
# ---------------------------------------------------------------------------


@dataclass
class APIExplorer:
    """Generates an interactive HTML/CSS API explorer page.

    The generated page is a fully self-contained HTML document using only
    inline CSS and vanilla JavaScript (no external CDN dependencies).  It
    lists all MANIFOLD endpoints with collapsible request/response examples
    and a simple "Try It" form.

    Parameters
    ----------
    endpoints:
        List of :class:`APIEndpoint` descriptors to render.
    module_docs:
        Optional list of :class:`ModuleDoc` objects to render in a
        "Code Reference" section.
    title:
        Page title.  Default: ``"MANIFOLD API Explorer"``.
    version:
        Version string shown in the header.  Default: ``"3.0.0"``.

    Example
    -------
    ::

        explorer = APIExplorer(endpoints=MANIFOLD_ENDPOINTS)
        html = explorer.render()
        # Serve html via the HTTP server
    """

    endpoints: list[APIEndpoint] = field(default_factory=lambda: list(MANIFOLD_ENDPOINTS))
    module_docs: list[ModuleDoc] = field(default_factory=list)
    title: str = "MANIFOLD API Explorer"
    version: str = "3.0.0"

    def render(self) -> str:
        """Generate and return the complete HTML page.

        Returns
        -------
        str
            Self-contained HTML string.
        """
        endpoint_cards = self._render_endpoint_cards()
        module_section = self._render_module_section()
        now_utc = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>{self.title}</title>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;padding:1.5rem}}
    h1{{font-size:1.8rem;font-weight:700;color:#fff}}
    h2{{font-size:1.2rem;font-weight:600;color:#94a3b8;margin:2rem 0 1rem}}
    h3{{font-size:1rem;font-weight:600;color:#e2e8f0}}
    .badge{{display:inline-block;padding:0.15rem 0.5rem;border-radius:999px;font-size:0.7rem;font-weight:700;letter-spacing:0.05em}}
    .get{{background:#1d4ed8;color:#bfdbfe}}
    .post{{background:#065f46;color:#a7f3d0}}
    .card{{background:#1e293b;border-radius:0.75rem;padding:1.25rem;margin-bottom:1rem;border:1px solid #334155}}
    .card:hover{{border-color:#475569}}
    .tag{{display:inline-block;background:#1e3a5f;color:#93c5fd;border-radius:0.25rem;
           padding:0.1rem 0.4rem;font-size:0.65rem;margin-right:0.25rem;margin-top:0.25rem}}
    summary{{cursor:pointer;color:#94a3b8;font-size:0.8rem;margin-top:0.5rem;padding:0.25rem 0}}
    summary:hover{{color:#e2e8f0}}
    pre{{background:#0f172a;border-radius:0.5rem;padding:0.75rem;font-size:0.72rem;
         color:#86efac;overflow-x:auto;white-space:pre-wrap;margin-top:0.5rem}}
    .path{{font-family:monospace;font-size:1rem;color:#38bdf8;margin-left:0.5rem}}
    .summary-text{{color:#cbd5e1;font-size:0.875rem;margin-top:0.4rem}}
    input,textarea{{width:100%;background:#0f172a;color:#e2e8f0;border:1px solid #334155;
                    border-radius:0.375rem;padding:0.5rem;font-family:monospace;font-size:0.8rem;
                    margin-top:0.25rem;resize:vertical}}
    button{{background:#1d4ed8;color:#fff;border:none;border-radius:0.375rem;
            padding:0.4rem 1rem;cursor:pointer;font-size:0.8rem;margin-top:0.5rem}}
    button:hover{{background:#2563eb}}
    .response-box{{margin-top:0.5rem;min-height:2rem}}
    .mod-entry{{border-left:2px solid #334155;padding-left:1rem;margin-bottom:0.5rem}}
    .mod-name{{color:#a78bfa;font-family:monospace;font-size:0.85rem}}
    .entry-sig{{color:#38bdf8;font-family:monospace;font-size:0.75rem}}
    .entry-doc{{color:#94a3b8;font-size:0.75rem;margin-top:0.15rem}}
    .flex-row{{display:flex;align-items:center;gap:0.5rem}}
    .kpi{{background:#1e293b;border-radius:0.5rem;padding:0.75rem 1.25rem;
          text-align:center;flex:1;border:1px solid #334155}}
    .kpi-val{{font-size:1.5rem;font-weight:700;color:#38bdf8}}
    .kpi-label{{font-size:0.7rem;color:#64748b;margin-top:0.2rem}}
    footer{{color:#334155;font-size:0.75rem;text-align:center;margin-top:2rem}}
  </style>
</head>
<body>
  <div style="max-width:900px;margin:0 auto">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.5rem">
      <div>
        <h1>⚡ {self.title}</h1>
        <p style="color:#64748b;font-size:0.85rem;margin-top:0.3rem">
          MANIFOLD v{self.version} &nbsp;·&nbsp; Pure-Python Zero-Dependency Trust OS
          &nbsp;·&nbsp; {len(self.endpoints)} endpoints &nbsp;·&nbsp; {len(self.module_docs)} modules documented
        </p>
      </div>
      <span style="color:#475569;font-size:0.75rem">{now_utc}</span>
    </div>

    <!-- KPI Strip -->
    <div style="display:flex;gap:0.75rem;margin-bottom:1.5rem">
      <div class="kpi"><div class="kpi-val">{len(self.endpoints)}</div><div class="kpi-label">Endpoints</div></div>
      <div class="kpi"><div class="kpi-val">{len([e for e in self.endpoints if e.method=='POST'])}</div><div class="kpi-label">POST</div></div>
      <div class="kpi"><div class="kpi-val">{len([e for e in self.endpoints if e.method=='GET'])}</div><div class="kpi-label">GET</div></div>
      <div class="kpi"><div class="kpi-val">{len(self.module_docs)}</div><div class="kpi-label">Modules</div></div>
    </div>

    <h2>🔌 Endpoints</h2>
    {endpoint_cards}

    {module_section}

    <footer>Generated by MANIFOLD Phase 55 — Self-Documenting API · Zero external dependencies</footer>
  </div>

  <script>
    function tryEndpoint(path, method, bodyId, respId) {{
      var bodyEl = document.getElementById(bodyId);
      var respEl = document.getElementById(respId);
      var bodyText = bodyEl ? bodyEl.value : '';
      respEl.innerHTML = '<pre style="color:#94a3b8">⏳ Sending ' + method + ' ' + path + '…</pre>';
      var opts = {{method: method, headers: {{'Content-Type':'application/json'}}}};
      if (method !== 'GET' && bodyText.trim()) opts.body = bodyText;
      fetch(path, opts)
        .then(function(r){{return r.text().then(function(t){{return {{status:r.status,text:t}}}})}})
        .then(function(r){{
          var col = r.status < 300 ? '#86efac' : '#f87171';
          var pretty = '';
          try{{pretty = JSON.stringify(JSON.parse(r.text), null, 2)}}catch(e){{pretty = r.text.substring(0,2000)}}
          respEl.innerHTML = '<pre style="color:' + col + '">HTTP ' + r.status + '\\n' + pretty + '</pre>';
        }})
        .catch(function(e){{respEl.innerHTML = '<pre style="color:#f87171">Error: ' + e.message + '</pre>';}});
    }}
  </script>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Private rendering helpers
    # ------------------------------------------------------------------

    def _render_endpoint_cards(self) -> str:
        """Render all endpoint cards as HTML."""
        cards: list[str] = []
        for i, ep in enumerate(self.endpoints):
            badge_cls = "get" if ep.method == "GET" else "post"
            tags_html = "".join(f"<span class='tag'>{t}</span>" for t in ep.tags)
            uid = f"ep{i}"

            # Try-It form
            if ep.method == "POST" and ep.request_example:
                body_area = f"""
              <div style="margin-top:0.5rem">
                <div style="font-size:0.75rem;color:#64748b">Request body (JSON):</div>
                <textarea id="{uid}_body" rows="4">{ep.request_example}</textarea>
              </div>"""
            else:
                body_area = f'<input type="hidden" id="{uid}_body" value=""/>'

            try_it_btn = (
                f'<button onclick="tryEndpoint(\'{ep.path}\',\'{ep.method}\','
                f'\'{uid}_body\',\'{uid}_resp\')">▶ Try It</button>'
            )

            resp_example_html = ""
            if ep.response_example:
                resp_example_html = (
                    f"<details><summary>Response example</summary>"
                    f"<pre>{ep.response_example}</pre></details>"
                )

            cards.append(f"""
  <div class="card">
    <div class="flex-row">
      <span class="badge {badge_cls}">{ep.method}</span>
      <span class="path">{ep.path}</span>
    </div>
    <div class="summary-text">{ep.summary}</div>
    {tags_html}
    {'<p style="color:#64748b;font-size:0.8rem;margin-top:0.5rem">' + ep.description + '</p>' if ep.description else ''}
    {resp_example_html}
    {body_area}
    {try_it_btn}
    <div class="response-box" id="{uid}_resp"></div>
  </div>""")

        return "\n".join(cards)

    def _render_module_section(self) -> str:
        """Render the module documentation section."""
        if not self.module_docs:
            return ""

        mod_cards: list[str] = []
        for md in self.module_docs:
            short_doc = (md.module_docstring or "")[:200].replace("\n", " ")
            entries_html = ""
            for entry in md.entries[:20]:  # limit to 20 per module
                entries_html += (
                    f"<div class='mod-entry'>"
                    f"<span class='badge {'get' if entry.kind == 'class' else 'post'}' "
                    f"style='font-size:0.6rem'>{entry.kind}</span> "
                    f"<span class='entry-sig'>{entry.signature[:120]}</span>"
                    + (f"<div class='entry-doc'>{entry.docstring[:200]}</div>" if entry.docstring else "")
                    + "</div>"
                )
            mod_cards.append(f"""
  <div class="card">
    <div class="flex-row">
      <span class="mod-name">{md.module_name}</span>
      <span style="color:#64748b;font-size:0.75rem">{len(md.public_classes)} classes · {len(md.public_functions)} functions</span>
    </div>
    {f'<div style="color:#64748b;font-size:0.8rem;margin-top:0.4rem">{short_doc}</div>' if short_doc else ''}
    <details style="margin-top:0.5rem">
      <summary>{len(md.entries)} public symbols</summary>
      <div style="margin-top:0.5rem">{entries_html}</div>
    </details>
  </div>""")

        return f"<h2>📚 Code Reference</h2>\n" + "\n".join(mod_cards)

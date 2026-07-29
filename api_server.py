"""
EasyAgent API Server - FastAPI based
Provides stable REST API and simple Web UI
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from easy_agent import EasyAgent, ToolSystem

# =========================================
# FastAPI App
# =========================================
app = FastAPI(
    title="EasyAgent API",
    description="AMD Radeon GPU Intelligent Agent System - REST API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = None

# =========================================
# Request/Response Models
# =========================================
class QueryRequest(BaseModel):
    query: str
    max_steps: Optional[int] = 5

class QueryResponse(BaseModel):
    success: bool
    answer: str
    steps: List[Dict]
    tools_used: List[str]

class ToolRequest(BaseModel):
    tool_name: str
    arguments: Dict

class ToolResponse(BaseModel):
    success: bool
    result: str

class SystemInfoResponse(BaseModel):
    success: bool
    info: Dict

# =========================================
# Startup
# =========================================
@app.on_event("startup")
async def startup_event():
    global agent
    print("Loading EasyAgent model...")
    model_path = "./models/Qwen2.5-7B-Instruct"
    agent = EasyAgent(model_path)
    print("EasyAgent ready!")

# =========================================
# API Endpoints
# =========================================
@app.get("/")
async def root():
    """Redirect to API documentation"""
    return HTMLResponse(content="""
    <html>
        <head>
            <title>EasyAgent API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
                h1 { color: #2c3e50; }
                .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .method { color: #27ae60; font-weight: bold; }
                a { color: #3498db; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .btn { display: inline-block; padding: 10px 20px; background: #3498db; color: white;
                       border-radius: 5px; margin: 5px; }
                .btn:hover { background: #2980b9; }
            </style>
        </head>
        <body>
            <h1>🤖 EasyAgent API Server</h1>
            <p>AMD Radeon GPU Intelligent Agent System</p>

            <h2>📚 Documentation</h2>
            <a href="/docs" class="btn">Swagger UI</a>
            <a href="/redoc" class="btn">ReDoc</a>

            <h2>🎮 Web Interface</h2>
            <a href="/web" class="btn">Simple Web UI</a>

            <h2>🔌 API Endpoints</h2>
            <div class="endpoint">
                <span class="method">POST</span> <code>/api/query</code> - Submit a query to the agent
            </div>
            <div class="endpoint">
                <span class="method">POST</span> <code>/api/tool</code> - Execute a specific tool
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <code>/api/tools</code> - List available tools
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <code>/api/system</code> - Get system information
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <code>/health</code> - Health check
            </div>
        </body>
    </html>
    """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": agent is not None}

@app.get("/api/tools")
async def list_tools():
    """List all available tools"""
    tools = []
    for name, tool in agent.tools.tools.items():
        tools.append({
            "name": name,
            "description": tool["description"],
            "parameters": tool["schema"]
        })
    return {"success": True, "tools": tools}

@app.get("/api/system", response_model=SystemInfoResponse)
async def get_system_info():
    """Get system information"""
    try:
        result = agent.tools.execute("system_info", {})
        # Parse the result into structured data
        info = {}
        for line in result.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()
        return {"success": True, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Submit a query to the EasyAgent

    The agent will:
    1. Analyze the query
    2. Plan execution steps
    3. Call tools as needed
    4. Return the final answer
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        # Collect execution trace
        steps = []
        tools_used = []

        def callback(info):
            steps.append(info)
            if info.get('type') == 'tool_call':
                tools_used.append(info.get('name', ''))

        # Run agent
        answer = agent.run(request.query, max_steps=request.max_steps, callback=callback)

        return {
            "success": True,
            "answer": answer,
            "steps": steps,
            "tools_used": tools_used
        }
    except Exception as e:
        return {
            "success": False,
            "answer": f"Error: {str(e)}",
            "steps": [],
            "tools_used": []
        }

@app.post("/api/tool", response_model=ToolResponse)
async def execute_tool(request: ToolRequest):
    """
    Execute a specific tool directly

    Use this to call tools without going through the agent reasoning loop
    """
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = agent.tools.execute(request.tool_name, request.arguments)
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "result": f"Error: {str(e)}"}

# =========================================
# Simple Web UI
# =========================================
@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """Simple Web UI using HTTP requests (no WebSocket)"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EasyAgent Web UI</title>
        <style>
            * { box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 20px;
            }
            .header h1 { margin: 0; font-size: 2em; }
            .header p { margin: 10px 0 0 0; opacity: 0.9; }
            .container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            .panel {
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .panel h2 {
                margin-top: 0;
                color: #333;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }
            textarea {
                width: 100%;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
                font-family: inherit;
                resize: vertical;
                min-height: 100px;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            button {
                background: #667eea;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                margin-top: 10px;
                width: 100%;
            }
            button:hover { background: #5568d3; }
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            .result {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 15px;
                min-height: 200px;
                max-height: 500px;
                overflow-y: auto;
                white-space: pre-wrap;
                font-size: 14px;
                line-height: 1.6;
            }
            .steps {
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 10px;
                margin: 10px 0;
                font-size: 13px;
            }
            .tool-call {
                background: #d1ecf1;
                border-left: 4px solid #17a2b8;
                padding: 10px;
                margin: 5px 0;
                font-size: 13px;
            }
            .examples {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 10px;
            }
            .example-btn {
                background: #e9ecef;
                border: 1px solid #dee2e6;
                padding: 8px 15px;
                border-radius: 20px;
                cursor: pointer;
                font-size: 13px;
                width: auto;
                margin: 0;
            }
            .example-btn:hover {
                background: #dee2e6;
            }
            .status {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 3px;
                font-size: 12px;
                margin-left: 10px;
            }
            .status.loading { background: #fff3cd; color: #856404; }
            .status.success { background: #d4edda; color: #155724; }
            .status.error { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 EasyAgent</h1>
            <p>AMD Radeon GPU Intelligent Agent System</p>
        </div>

        <div class="container">
            <div class="panel">
                <h2>💬 Query Agent</h2>
                <textarea id="query" placeholder="Enter your query here...&#10;&#10;Examples:&#10;- Get current system time and GPU status&#10;- Search knowledge about AMD GPU&#10;- Calculate the sum of first 100 primes"></textarea>

                <div class="examples">
                    <button class="example-btn" onclick="setQuery('Get current system information')">System Info</button>
                    <button class="example-btn" onclick="setQuery('Search knowledge about ROCm')">ROCm Knowledge</button>
                    <button class="example-btn" onclick="setQuery('Calculate 2^10 + 3^5')">Calculate</button>
                    <button class="example-btn" onclick="setQuery('What is the current time?')">Time Query</button>
                </div>

                <button id="submitBtn" onclick="submitQuery()">Submit Query</button>

                <div id="queryStatus" class="status" style="display:none;"></div>
            </div>

            <div class="panel">
                <h2>📊 Result</h2>
                <div id="result" class="result">Submit a query to see results here...</div>
            </div>
        </div>

        <div style="margin-top: 20px; text-align: center; color: #666; font-size: 12px;">
            <p>API Documentation: <a href="/docs">/docs</a> | <a href="/redoc">/redoc</a></p>
        </div>

        <script>
            function setQuery(text) {
                document.getElementById('query').value = text;
            }

            function showStatus(message, type) {
                const status = document.getElementById('queryStatus');
                status.textContent = message;
                status.className = 'status ' + type;
                status.style.display = 'inline-block';
            }

            async function submitQuery() {
                const query = document.getElementById('query').value.trim();
                if (!query) {
                    alert('Please enter a query');
                    return;
                }

                const submitBtn = document.getElementById('submitBtn');
                const resultDiv = document.getElementById('result');

                submitBtn.disabled = true;
                submitBtn.textContent = 'Processing...';
                showStatus('Running agent...', 'loading');
                resultDiv.textContent = 'Processing your query...\\n\\nThis may take a few seconds as the agent reasons and executes tools.';

                try {
                    const response = await fetch('/api/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: query, max_steps: 5 })
                    });

                    const data = await response.json();

                    let output = '';

                    // Show steps
                    if (data.steps && data.steps.length > 0) {
                        output += '=== Execution Trace ===\\n\\n';
                        data.steps.forEach((step, i) => {
                            if (step.type === 'task') {
                                output += `[Step ${i+1}] Task: ${step.content}\\n`;
                            } else if (step.type === 'tool_call') {
                                output += `[Step ${i+1}] Calling tool: ${step.name}\\n`;
                                output += `  Arguments: ${JSON.stringify(step.args)}\\n`;
                            } else if (step.type === 'tool_result') {
                                output += `[Step ${i+1}] Tool result: ${step.result.substring(0, 100)}...\\n`;
                            } else if (step.type === 'answer') {
                                output += `[Step ${i+1}] Generated answer\\n`;
                            }
                        });
                        output += '\\n';
                    }

                    // Show tools used
                    if (data.tools_used && data.tools_used.length > 0) {
                        output += `=== Tools Used ===\\n${data.tools_used.join(', ')}\\n\\n`;
                    }

                    // Show final answer
                    output += `=== Final Answer ===\\n\\n${data.answer}`;

                    resultDiv.textContent = output;

                    if (data.success) {
                        showStatus('Success!', 'success');
                    } else {
                        showStatus('Error occurred', 'error');
                    }

                } catch (error) {
                    resultDiv.textContent = `Error: ${error.message}`;
                    showStatus('Request failed', 'error');
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Submit Query';
                }
            }

            // Allow Ctrl+Enter to submit
            document.getElementById('query').addEventListener('keydown', function(e) {
                if (e.ctrlKey && e.key === 'Enter') {
                    submitQuery();
                }
            });
        </script>
    </body>
    </html>
    """, status_code=200)

# =========================================
# Main
# =========================================
if __name__ == "__main__":
    print("Starting EasyAgent API Server...")
    print("Web UI: http://127.0.0.1:8000/web")
    print("API Docs: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)

VisionIQ Development Guide
1. Overview
Let's start with the big picture. VisionIQ is an image quality analysis application. You upload an image, and it evaluates that image for degradation and identifies any potential defects.

Under the hood, we use a smart, hierarchical ML pipeline:

Stage 1: A binary classifier that figures out if an image is "clean" or "degraded."

Stage 2: A multi-class classifier that pinpoints what kind of degradation it is (blur, noise, etc.).

Decision layer: To avoid blindly trusting the ML, we also have some computer-vision rules that can override uncertain ML predictions.

Quality score: We calculate a weighted penalty based on the issues we detect, giving you a final score from 0 to 100.

The stack is:

FastAPI (Python) — for the backend API and ML pipeline.

React + Vite (JavaScript) — for the frontend user interface.

SQLite — for storing a local history of your analyses.

Docker + Docker Compose — the recommended way to run the project locally.

Nginx — serves the frontend and proxies API requests to the backend.

2. Prerequisites
The easiest way to get started is using Docker. Here's what you'll need installed:

Git — to clone the repository.

Docker — the container platform.

Docker Compose — version 2.0 or higher, for orchestrating the services.

A quick note: The repo uses the Docker Compose V2 plugin format (docker compose, with a space), not the old docker-compose (with a hyphen). If you have the old one, it might still work, but the modern format is what we've tested.

We've verified these minimum versions:

Docker: 20.10+

Docker Compose: 2.0+

Node: 20 (used inside the frontend Dockerfile)

Python: 3.12 (used inside the backend Dockerfile)

(No need to install Node or Python on your host if you're using Docker—they're already handled inside the containers!)

3. Clone the Repository
Alright, let's get the code onto your machine:

bash
git clone https://github.com/your-org/visioniq.git
cd visioniq
If your repository is hosted somewhere else, just swap out the URL above.

4. Run Locally with Docker
This is the main, recommended way to run VisionIQ. Just run:

bash
docker compose up --build
What happens when you run this?
This single command builds and spins up two services defined in docker-compose.yml:

Service	Role	Ports
backend	The FastAPI app + ML pipeline	Internal port 8000; exposed at http://localhost/api/ via Nginx
frontend	The React app served by Nginx	Port 80 on the host
Let's peek at the services
backend: Built from the root Dockerfile. It runs on Python 3.12, installs the stuff in requirements.txt, and starts up uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000}. It uses DATABASE_URL: sqlite:////app/data/visioniq.db, and we mount a local ./data folder so your database persists on your hard drive. It listens on internal port 8000.

frontend: Built from frontend/Dockerfile in a multi-stage setup. First, a Node 20 Alpine image compiles the React Vite app. Then, a lean nginx:alpine image serves the built static files. It's mapped to port 80 on your machine.

How does the frontend talk to the backend?
When you visit http://localhost in your browser, the frontend makes API calls to /api/... routes. Nginx (inside the frontend container) hears those calls and proxies them to the backend service using the internal Docker name backend. Here's the config line that does the magic:

text
location /api/ {
    proxy_pass http://backend:10000;
    ...
}
Database
Tech: SQLite (just a file, nice and simple!)

Location: visioniq.db

Inside Docker: /app/data/visioniq.db

Volume: The ./data:/app/data bind mount in the compose file connects that database file to your host.

Persistence: As long as you don't delete the ./data folder on your host, your analysis history will survive docker compose down and docker compose up.

Environment variables
The docker-compose.yml sets this variable for you:

Variable	Value	Source
DATABASE_URL	sqlite:////app/data/visioniq.db	Hardcoded in docker-compose.yml
You don't need to create a .env file for the Docker workflow. The Dockerfile just reads PORT from the environment (defaulting to 10000).

5. Access the Application
Once everything is running, here's where to go:

Service	URL
Frontend	http://localhost
Backend API	http://localhost:8000 (if you need direct access)
Backend health	http://localhost:8000/health
API docs (Swagger)	http://localhost:8000/docs
Health check
You can verify the backend is alive with:

bash
curl http://localhost:8000/health
You should see this response:

json
{"status": "healthy", "timestamp": "..."}
API endpoints available
GET / — root endpoint: {"message": "VisionIQ API is running!"}

GET /health — health check

POST /api/v1/analyze — upload an image for analysis

GET /api/v1/analyses — list your analysis history

GET /api/v1/analyses/{id} — get a specific analysis by ID

POST /api/v1/batch — batch image analysis

6. Verify the Installation
After you run docker compose up --build, let's make sure everything is working:

bash
# Check if containers are running
docker compose ps

# Check the backend is healthy
curl http://localhost:8000/health

# Check the frontend is accessible
curl -s -o /dev/null -w "%{http_code}" http://localhost
# Expected: 200
If you see all services "Up" and the health endpoint returns a healthy status, you're good to go!

7. Testing Image Analysis
Via the frontend (the easy way)
Make sure services are running (docker compose ps shows them as Up).

Open http://localhost in your browser.

Use the upload area to pick an image file.

Supported formats are JPG, PNG, WEBP, and BMP.

Click Analyze Image.

Wait a moment for the analysis to complete.

The result card will show you:

Quality score (0–100)

Clean vs degraded probability

Stage 1 and Stage 2 predictions

Detected issue type and confidence

A list of issues with severity

Via curl (direct API access)
If you prefer the command line:

bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
The response is a neat JSON blob with the same fields the frontend shows.

8. Project Structure
Let's take a quick tour of the folders:

text
VisionIQ/
├── Dockerfile                  # Root: Python/FastAPI builder
├── docker-compose.yml          # Orchestrates backend + frontend
├── database.py                 # SQLAlchemy + SQLite setup
├── requirements.txt            # Python dependencies
├── .env.example              # (empty — no vars required for Docker)
├── .gitignore                # OS/IDE ignores
├── api.py                      # FastAPI entrypoint with routes
├── render.yaml               # Render.com deployment config
├── visioniq.db                 # SQLite database file
├── frontend/                   # React + Vite application
│   ├── Dockerfile              # Node 20 Alpine multi-stage builder
│   ├── nginx/
│   │   └── default.conf        # Nginx: serves UI + proxies /api/
│   ├── package.json            # Frontend dependencies
│   └── vite.config.js          # Vite React plugin config
├── backend/                    # Python package
│   └── app/                    # FastAPI app modules
│       ├── api.py              # App routes (/health, /analyze, etc.)
│       ├── database.py         # DB setup
│       ├── models.py           # SQLAlchemy models
│       └── features/           # Feature extractors
├── ml/                         # ML pipeline artifacts
│   ├── artifacts/              # Trained RF / label encoders
│   ├── train.py                # Training script
│   └── evaluate*.py            # Evaluation scripts
├── dataset/                    # Image datasets
├── quality/                    # Decision logic
├── tests/                      # pytest tests
└── DEVELOPMENT.md            # This file
9. Architecture
Here's a simple visual of how data flows:

text
+----------------+      +------------------+      +---------------------+
|   Browser/CLI  | -->  |   Nginx (frontend)| -->  |   FastAPI Backend   |
+----------------+      +------------------+      +---------------------+
          |                        |   |               |
          |          /api/        |   |               |
          v                        v   v               v
    http://localhost        port 80    http://backend:10000  SQLite (visioniq.db)
                                      inside container
The React app lives inside Nginx on port 80. Nginx forwards any /api/ requests to the FastAPI service on the internal network. The backend runs the ML pipeline and stores results in the SQLite database. The model artifacts sit in ml/artifacts/ and get loaded when the server starts.

10. Environment Variables
Here are the variables the project uses:

Variable	Required?	Default/Value	Description	Consumed by
DATABASE_URL	No (Docker)	sqlite:////app/data/visioniq.db	SQLite database connection string	Backend (sqlalchemy)
PORT	No (Docker)	10000	Uvicorn port	Root Dockerfile CMD
FRONTEND_URL	No (Docker)	Not set in compose	Frontend URL for CORS	Root api.py CORS middleware
FRONTEND_URL	Yes (Render)	From service property	Frontend URL for Render deployment	Render render.yaml
For local development, you don't need to worry about a .env file. The compose file handles the DATABASE_URL, and the port mapping is automatic. If you do create a .env file, it might override the compose variables, so just use the compose file as-is.

11. Database
Tech: SQLite (a simple file-based database)

Location: visioniq.db at the repo root; inside Docker, it's at /app/data/visioniq.db (bind-mounted from your host's ./data folder).

Persistence: The ./data:/app/data volume ensures your history survives container restarts. As long as you don't delete the ./data folder, your data is safe.

Reset: If you want to wipe all analysis data, remove the visioniq.db file and the ./data directory. If you also want to clear any named volumes, run docker compose down -v. Warning: This deletes the database!

Not destructive by default: Running docker compose down alone will NOT delete your database—the ./data bind mount is preserved.

12. Useful Docker Commands
Command	Description
docker compose up --build	Build (if needed) and start all services
docker compose up -d	Start everything in the background (detached mode)
docker compose ps	See running containers and their status
docker compose logs	Show logs from all services
docker compose logs -f backend	Follow backend logs in real-time
docker compose logs -f frontend	Follow frontend (Nginx) logs in real-time
docker compose down	Stop and remove containers, networks, and volumes (safe for your data)
docker compose down -v	Stop services AND remove named volumes + the bind-mounted ./data directory (destroys database)
docker compose restart	Restart all services
docker compose build	Rebuild images without starting containers
13. Development Workflow
Clone: git clone <repository> and cd visioniq.

Start: docker compose up --build.

Make changes:

Frontend: Edit files in frontend/src/. Since the Docker image serves static files, you'll need to rebuild (docker compose up --build) or run the dev server locally (see section 14).

Backend: Edit files in backend/app/ or the root api.py. You'll need to rebuild the Docker image to pick up changes.

Test: Open http://localhost, upload an image, and see the results.

Check logs: Use docker compose logs -f backend or frontend to debug.

Commit: git add ..., git commit -m "...", git push.

14. Frontend Development
Framework: React 19.2.8

Build tool: Vite 8 with @vitejs/plugin-react

Package manager: npm

Source files: frontend/src/

Build: npm run build (produces dist/ assets)

Dev mode: npm run vite (runs at http://localhost:5173)

If you want to work on the frontend without Docker (to get hot-reload):

bash
cd frontend
npm install       # install dependencies (once)
npm run dev       # start Vite dev server at http://localhost:5173
You'll need to make sure the backend is running. You might need to set a VITE_API_BASE in a .env file at the frontend root to point to your backend.

15. Backend Development
Python version: 3.12 (in the root Dockerfile)

FastAPI entrypoint: api.py

Dependencies: requirements.txt

Starting the backend: The Dockerfile runs uvicorn api:app --host 0.0.0.0 --port ${PORT:-10000}.

If you want to run it outside Docker:

bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
The backend will be available at http://localhost:8000, and you can view the Swagger docs at http://localhost:8000/docs.

16. API Development
Important API routes
Route	Method	Description
/	GET	Root endpoint: {"message": "VisionIQ API is running!"}
/health	GET	Health check with timestamp
/api/v1/analyze	POST	Upload image for single analysis
/api/v1/analyses	GET	List analysis history (paginated)
/api/v1/analyses/{id}	GET	Retrieve specific analysis by ID
/api/v1/batch	POST	Upload multiple images for batch analysis
Example: Single image upload
bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/test-image.jpg"
Example: Get analysis history
bash
curl -s http://localhost:8000/api/v1/analyses | jq .
Example: Health check
bash
curl -s http://localhost:8000/health | jq .
17. Troubleshooting
A few common issues you might run into:

Problem	Likely cause	Fix
Frontend container keeps restarting	Nginx config error or port conflict	Check docker compose logs -f frontend; verify port 80 is free
Backend not reachable at http://localhost:8000	Backend container not running or port mapped incorrectly	Run docker compose ps; check docker compose logs -f backend for errors
curl http://localhost:8000/health fails	Backend not built/started	Run docker compose up --build
Analysis fails (500 error)	Missing ML model artifacts	Ensure ml/artifacts/ exists with the .joblib files
Nginx 502 Bad Gateway	Backend not responding on backend:10000	Check backend is healthy; verify networks
Database not persisting	./data directory deleted	Do not remove the ./data folder!
CORS errors in browser	FRONTEND_URL not configured	Set FRONTEND_URL via .env or in Render dashboard
Frontend shows blank screen on refresh	Nginx try_files not matching	Run docker compose up --build to rebuild static files
18. Clean Reset
To completely restart everything:

bash
docker compose down -v
Warning: This deletes your ./data folder and the SQLite database. All history is lost!

To keep the database but restart the containers:

bash
docker compose down
docker compose up --build
To just restart:

bash
docker compose restart
19. Production vs Local Development
The repo comes with a render.yaml for deploying to Render.com. Here are the main differences:

Aspect	Local Docker Compose	Production (Render)
Services	backend + frontend (Nginx)	Separate Render services
Database	SQLite file in ./data	External database (Render provides PostgreSQL on paid plans)
Domain	http://localhost	Custom domain or onrender.com
HTTPS	Not configured	Automatic HTTPS on Render
CORS	Relaxed	Restricted to production frontend URL
Env vars	Set in docker-compose.yml	Set in Render dashboard or render.yaml
For local dev, stick with Docker Compose. For production, use Render.

20. Contributing
If you want to contribute to the project:

Create a branch: git checkout -b feature/amazing-feature

Make changes: Edit files in backend/app/ or frontend/src/

Test locally: Run docker compose up --build, then test at http://localhost

Check status: git diff and git status

Commit: git commit -m "Add amazing feature"

Push: git push origin feature/amazing-feature

Open PR: Push to your fork and submit a pull request on GitHub

Important: Always rebuild the Docker images after code changes!

Quick Start
bash
git clone https://github.com/your-org/visioniq.git
cd visioniq
docker compose up --build
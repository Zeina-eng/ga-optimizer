from fastapi import FastAPI, BackgroundTasks, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from backend.data_generator import generate_student_data
from backend.genetic_algorithm import GeneticAlgorithm

app = FastAPI(
    title="ElasticNet GA Optimizer API",
    description="Backend service optimizing ElasticNet hyperparameters and feature selection using Genetic Algorithms.",
    version="1.0.0"
)

# -----------------------------
# Enable CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Global Application State
# -----------------------------
dataset = None
ga = None
best_solution = None
running = False
current_generation = 0
total_generations = 0
progress_percent = 0.0
current_best_fitness = None


class GARunConfig(BaseModel):
    generations: int = Field(30, ge=1, le=200, description="Number of GA generations")
    population_size: int = Field(30, ge=2, le=200, description="Population size")
    crossover_rate: float = Field(0.8, ge=0.1, le=1.0, description="Crossover probability")
    mutation_rate: float = Field(0.1, ge=0.01, le=0.5, description="Mutation probability")
    gamma: float = Field(0.02, ge=0.0, le=0.5, description="Feature count penalty multiplier")



# -----------------------------
# Generate Dataset Endpoint
# -----------------------------
@app.post("/generate-data")
def generate_data(samples: int = Query(500, ge=50, le=5000)):
    global dataset, ga, best_solution, running

    if running:
        raise HTTPException(status_code=400, detail="Cannot regenerate dataset while GA optimization is running.")

    dataset = generate_student_data(num_students=samples)
    ga = None
    best_solution = None

    return {
        "message": "Dataset generated successfully",
        "samples": len(dataset),
        "columns": list(dataset.columns),
    }


# -----------------------------
# Background GA Worker
# -----------------------------
def run_ga_task(config: GARunConfig):
    global ga, dataset, best_solution, running
    global current_generation, total_generations, progress_percent, current_best_fitness

    running = True
    current_generation = 0
    total_generations = config.generations
    progress_percent = 0.0
    current_best_fitness = None

    def on_progress(gen, total_gen, best_fit, info):
        global current_generation, total_generations, progress_percent, current_best_fitness
        current_generation = gen
        total_generations = total_gen
        progress_percent = round((gen / total_gen) * 100, 1)
        current_best_fitness = round(best_fit, 4)

    try:
        ga = GeneticAlgorithm(
            dataframe=dataset,
            population_size=config.population_size,
            generations=config.generations,
            crossover_rate=config.crossover_rate,
            mutation_rate=config.mutation_rate,
            gamma=config.gamma,
        )

        best_solution = ga.evolve(progress_callback=on_progress)
    finally:
        running = False
        progress_percent = 100.0


# -----------------------------
# Start GA Endpoint
# -----------------------------
@app.post("/run-ga")
def start_ga(background_tasks: BackgroundTasks, config: Optional[GARunConfig] = Body(None)):
    global dataset, running

    if dataset is None:
        raise HTTPException(status_code=400, detail="Please generate or load a dataset first.")

    if running:
        raise HTTPException(status_code=409, detail="Genetic Algorithm optimization is already running.")

    if config is None:
        config = GARunConfig()

    background_tasks.add_task(run_ga_task, config)

    return {
        "message": "Genetic Algorithm optimization started.",
        "config": config.model_dump()
    }


# -----------------------------
# Current Status Endpoint
# -----------------------------
@app.get("/status")
def status():
    global running, current_generation, total_generations, progress_percent, current_best_fitness

    return {
        "running": running,
        "current_generation": current_generation,
        "total_generations": total_generations,
        "progress_percent": progress_percent,
        "current_best_fitness": current_best_fitness,
    }


# -----------------------------
# Best Result Endpoint
# -----------------------------
@app.get("/best")
def best():
    global best_solution

    if best_solution is None:
        return {
            "error": "No optimization has been completed yet."
        }

    return best_solution


# -----------------------------
# Fitness History Endpoint
# -----------------------------
@app.get("/history")
def history():
    global ga

    if ga is None or not hasattr(ga, "history"):
        return []

    return ga.history


# -----------------------------
# Dataset Preview Endpoint
# -----------------------------
@app.get("/dataset")
def dataset_preview(limit: int = Query(20, ge=1, le=100)):
    global dataset

    if dataset is None:
        return {"samples": 0, "columns": [], "data": []}

    return {
        "samples": len(dataset),
        "columns": list(dataset.columns),
        "data": dataset.head(limit).to_dict(orient="records")
    }


# -----------------------------
# Root & Static Files
# -----------------------------
import os
from fastapi.staticfiles import StaticFiles

frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")


@app.get("/")
def root():
    return {
        "name": "ElasticNet GA Optimizer API",
        "status": "online",
        "app_url": "/app/",
        "docs_url": "/docs"
    }
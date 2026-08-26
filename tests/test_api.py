import unittest
from fastapi.testclient import TestClient

from backend.main import app


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")

    def test_dataset_generation(self):
        response = self.client.post("/generate-data?samples=100")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["samples"], 100)
        self.assertIn("Final_Marks", data["columns"])

    def test_dataset_preview(self):
        self.client.post("/generate-data?samples=100")
        response = self.client.get("/dataset?limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["samples"], 100)
        self.assertEqual(len(data["data"]), 10)

    def test_run_ga_and_status(self):
        self.client.post("/generate-data?samples=100")

        config = {
            "generations": 3,
            "population_size": 10,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "gamma": 0.02
        }

        response = self.client.post("/run-ga", json=config)
        self.assertEqual(response.status_code, 200)

        status_resp = self.client.get("/status")
        self.assertEqual(status_resp.status_code, 200)
        status_data = status_resp.json()
        self.assertIn("running", status_data)


if __name__ == "__main__":
    unittest.main()

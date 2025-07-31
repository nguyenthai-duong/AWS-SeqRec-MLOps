from locust import HttpUser, task, between

class InferenceUser(HttpUser):
    wait_time = between(0.0001,0.0002)  # thời gian chờ giữa các request (giây)

    @task
    def infer(self):
        user_id = "AFI4TKPAEMA6VBRHQ25MUXLHEIBA"
        self.client.get(
            f"/infer?user_id={user_id}",
            headers={"accept": "application/json"}
        )

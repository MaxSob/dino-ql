from django.db import models

class TrainingRun(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    description = models.CharField(max_length=255, default="Training Run")

    def __str__(self):
        return f"Run {self.id} - {self.timestamp}"

class Episode(models.Model):
    run = models.ForeignKey(TrainingRun, on_delete=models.CASCADE, related_name='episodes')
    episode_number = models.IntegerField()
    score = models.IntegerField()
    time_alive = models.IntegerField()
    replay_data = models.JSONField() # Stores list of states/metrics for replay
    
    def __str__(self):
        return f"Ep {self.episode_number} (Score: {self.score})"

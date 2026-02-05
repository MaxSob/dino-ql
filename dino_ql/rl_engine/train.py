
import os
import sys
import django
import numpy as np
import argparse

# Setup Django Environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from dashboard.models import TrainingRun, Episode
from rl_engine.game import DinoGame
from rl_engine.agent import DQLAgent, QLAgent

def train(agent_type='dql', episodes=5000):
    game = DinoGame()
    state_size = 4
    action_size = 2
    
    if agent_type == 'ql':
        agent = QLAgent(state_size, action_size)
        description = f"QL (Tabular) Training Session - {episodes} episodes"
    else:
        agent = DQLAgent(state_size, action_size)
        description = f"DQL (Deep Q-Learning) Training Session - {episodes} episodes"
        
    batch_size = 32

    # Create a new Training Run
    run = TrainingRun.objects.create(description=description)
    print(f"Started Training Run: {run.id} using {agent_type.upper()} for {episodes} episodes")

    for e in range(episodes):
        state = game.reset()
        done = False
        score = 0
        
        # Determine if we should record this episode for replay
        # Record every 100th episode
        record_replay = (e % 100 == 0)
        replay_data = []

        while not done:
            action = agent.act(state)
            
            if record_replay:
                state_dict = game.get_state_dict()
                state_dict['action'] = int(action) # Make serializable
                replay_data.append(state_dict)

            next_state, reward, done = game.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score = game.score
            
            if done:
                print(f"episode: {e}/{episodes}, score: {score}, e: {agent.epsilon:.2f}, time: {game.time_alive}")
                
                if record_replay:
                    Episode.objects.create(
                        run=run,
                        episode_number=e,
                        score=score,
                        time_alive=game.time_alive,
                        replay_data=replay_data
                    )

            if agent_type == 'dql' and len(agent.memory) > batch_size:
                agent.replay(batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Dino QL Agent')
    parser.add_argument('--agent', type=str, default='dql', choices=['dql', 'ql'],
                        help='Agent type to train: dql (Deep Q-Learning) or ql (Tabular Q-Learning)')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of episodes to train for (default: 5000)')
    
    args = parser.parse_args()
    train(agent_type=args.agent, episodes=args.episodes)


from django.shortcuts import render
from django.http import JsonResponse
from .models import TrainingRun, Episode

def index(request):
    runs = TrainingRun.objects.all().order_by('-timestamp')
    run_id = request.GET.get('run_id')
    
    if run_id:
        try:
            latest_run = TrainingRun.objects.get(id=run_id)
        except TrainingRun.DoesNotExist:
             latest_run = runs.first()
    else:
        latest_run = runs.first()

    context = {
        'runs': runs,
        'latest_run': latest_run
    }
    return render(request, 'dashboard/index.html', context)

def api_run_data(request, run_id):
    try:
        run = TrainingRun.objects.get(id=run_id)
        episodes = run.episodes.all().order_by('episode_number')
        
        data = {
            'ids': [e.id for e in episodes],
            'episodes': [e.episode_number for e in episodes],
            'scores': [e.score for e in episodes],
            'time_alive': [e.time_alive for e in episodes],
            'has_replay': [bool(e.replay_data) for e in episodes]
        }
        return JsonResponse(data)
    except TrainingRun.DoesNotExist:
        return JsonResponse({'error': 'Run not found'}, status=404)

def api_episode_replay(request, episode_id):
    try:
        episode = Episode.objects.get(id=episode_id)
        return JsonResponse(episode.replay_data, safe=False)
    except Episode.DoesNotExist:
        return JsonResponse({'error': 'Episode not found'}, status=404)

def about(request):
    return render(request, 'dashboard/about.html')

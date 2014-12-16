from __future__ import division

from django.db.models import Count, Max, Min

from .models import Outcome, CuratedAnswer

def answer_group(model, key):
    if key == model.DISAGREE_ANSWER[0]:
        objs = (model.objects
                .annotate(ans_values=Count('answers__answer', distinct=True))
                .filter(ans_values__gt=1))
    elif key == model.UNSCREENED_ANSWER[0]:
        objs = model.objects.filter(answers__isnull=True)
    else:
        objs = (model.objects
            .filter(answers__isnull=False)
            .annotate(ans_max=Max('answers__answer'), ans_min=Min('answers__answer'))
            .filter(ans_max=key, ans_min=key))

    return objs

# def next_to_screen(user):
#     # one answer between all users
#     #done = CuratedAnswer.objects.filter(curator=user).values_list('outcomes', flat=True)
#     #return (Outcome.objects.order_by('experiment_id', 'collision_id').exclude(id__in=done).first())
#     # each user answers all
#     #return (Outcome.objects.order_by('experiment_id', 'collision_id').filter(answers__isnull=True, user__isnot=user).first())
#     # each user answers all, random order (avoid fatigue/runs)
#     return (Outcome.objects
#         .filter(answers__isnull=True, curator__isnot=user)
#         .order_by('?')
#         .first())

DEFAULT_ORDER = 'experiment_id', 'collision_id'

def next_to_screen(user, order=('?',)):
    # see if any aren't screened at all (by any user)
    obj = (Outcome.objects
            .order_by(*order)
            .filter(answers__isnull=True)
            .first())

    if obj is None:
        # if everything's done once, check what the user has left to do
        done = (CuratedAnswer.objects
                .filter(curator=user)
                .values_list('gap', flat=True))
        obj = (Outcome.objects
                .order_by(*order)
                .exclude(id__in=done)
                .first())

    return obj

def user_progress(user):
    n_done = CuratedAnswer.objects.filter(curator=user).count()
    n_total = Outcome.objects.count()

    return {
        'percent': 100 * (n_done / n_total),
        'done': n_done,
        'remaining': n_total - n_done,
        'total': n_total,
    }

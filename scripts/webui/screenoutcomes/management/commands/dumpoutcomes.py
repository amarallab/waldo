import sys
from collections import defaultdict

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from ...models import Outcome

def write_result_header():
    sys.stdout.write('eid,target,pA,pB,cA,cB,ans,n_ans\n')

def write_result_line(outcome):
    raise NotImplementedError('TODO. IS ALL WRONG.')
    answers = [ca.answer for ca in outcome.answers.all()]
    n_answers = len(answers)
    if not n_answers:
        return

    answers = set(answers)
    n_diff_ans = len(answers)

    if n_diff_ans > 1:
        answer = Outcome.DISAGREE_ANSWER
    elif n_diff_ans == 1:
        answer = answers.pop()

    sys.stdout.write(','.join(
            [str(x) for x in [
                    gap.experiment_id, gap.from_blob, gap.to_blob,
                    answer, n_answers]]) + '\n')


class Command(BaseCommand):
    help = 'Dump a CSV file of the collision results'

    def handle(self, *args, **options):
        outcomes = Outcome.objects.prefetch_related('answers').all()

        write_result_header()
        for outcome in outcomes:
            write_result_line(outcome)

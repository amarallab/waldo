import sys
from collections import defaultdict

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from ...models import Gap

def write_result_header():
    sys.stdout.write('eid,from_blob,to_blob,ans\n')

def write_result_line(gap, answer):
    sys.stdout.write(','.join([str(x) for x in [gap.experiment_id, gap.from_blob, gap.to_blob, answer]]) + '\n')


class Command(BaseCommand):
    help = 'Dump a CSV file of the collision results'

    def handle(self, *args, **options):
        gaps = Gap.objects.prefetch_related('answers').all()

        write_result_header()
        for gap in gaps:
            answers = set(ca.answer for ca in gap.answers.all())
            if len(answers) == 1:
                write_result_line(gap, answers.pop())
            elif len(answers) > 1:
                write_result_line(gap, 'disagreement')

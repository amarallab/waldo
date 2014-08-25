import sys
from collections import defaultdict

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from ...models import Collision

def write_result_line(collision, answer):
    sys.stdout.write(','.join([str(x) for x in [collision.experiment_id, collision.blob_id, answer]]) + '\n')


class Command(BaseCommand):
    help = 'Dump a CSV file of the collision results'

    def handle(self, *args, **options):
        collisions = Collision.objects.prefetch_related('curatedanswer_set').all()
        for collision in collisions:
            answers = set(ca.answer for ca in collision.curatedanswer_set.all())
            if len(answers) == 1:
                write_result_line(collision, answers.pop())
            elif len(answers) > 1:
                write_result_line(collision, 'xx')

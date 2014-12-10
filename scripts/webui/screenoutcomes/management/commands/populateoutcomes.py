import sys

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from ...models import Outcome


class Command(BaseCommand):
    help = 'Load images from the settings.OUTCOME_IMAGES folder'

    def handle(self, *args, **options):
        self.stdout.write('Searching in {}'.format(settings.OUTCOME_IMAGES.resolve()))
        already_loaded = 0
        added = 0
        to_add = []
        for fields in Outcome.fill():
            try:
                Outcome.objects.get(**fields)
                already_loaded += 1
            except Outcome.DoesNotExist:
                to_add.append(Outcome(**Outcome))
                added += 1
        Outcome.objects.bulk_create(to_add)

        print('Found {} total outcome records/images, added {} to database ({} already loaded)'
                .format(added + already_loaded, added, already_loaded))

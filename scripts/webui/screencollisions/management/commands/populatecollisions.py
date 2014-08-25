import sys

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from ...models import Collision


class Command(BaseCommand):
    help = 'Load images from the settings.COLLISION_IMAGES folder'

    def handle(self, *args, **options):
        self.stdout.write('Searching in {}'.format(settings.COLLISION_IMAGES.resolve()))
        already_loaded = 0
        added = 0
        to_add = []
        for eid, bid in Collision.find_images():
            try:
                Collision.objects.get(experiment_id=eid, blob_id=bid)
                already_loaded += 1
            except Collision.DoesNotExist:
                to_add.append(Collision(experiment_id=eid, blob_id=bid))
                added += 1
        Collision.objects.bulk_create(to_add)

        print('Found {} total images, added {} to database ({} already loaded)'
                .format(added + already_loaded, added, already_loaded))

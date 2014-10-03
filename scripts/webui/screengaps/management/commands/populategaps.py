import sys

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from ...models import Gap


class Command(BaseCommand):
    help = 'Load images from the settings.GAP_IMAGES folder'

    def handle(self, *args, **options):
        self.stdout.write('Searching in {}'.format(settings.GAP_IMAGES.resolve()))
        already_loaded = 0
        added = 0
        to_add = []
        for image_fields in Gap.find_images():
            try:
                Gap.objects.get(**image_fields)
                already_loaded += 1
            except Gap.DoesNotExist:
                to_add.append(Gap(**image_fields))
                added += 1
        Gap.objects.bulk_create(to_add)

        print('Found {} total gap images, added {} to database ({} already loaded)'
                .format(added + already_loaded, added, already_loaded))

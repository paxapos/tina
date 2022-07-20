from django.core.management.base import BaseCommand, CommandError
from ia.ia_engine import IaEngine

class Command(BaseCommand):

    help = 'Admin IA predictions'

    def add_arguments(self, parser):
        pass


    def handle(self, *args, **options):
        pass

    




from django.core.management.base import BaseCommand, CommandError
from ia.ia_engine import IaEngine

class Command(BaseCommand):

    help='Admin IA predictions'


    def add_arguments(self,parser):

        parser.add_argument('product', nargs=1,type=str)
        parser.add_argument('img', nargs=1,type=str)
        

    def handle(self,*args,** options):

        engine = IaEngine()
        ret=engine.predict(options['product'][0],['img'][0])

        self.stdout.write(self.style.SUCCESS('Successfully closed predict"%s"'%ret))

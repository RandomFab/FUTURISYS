import os
from dotenv import load_dotenv

ENV = os.getenv('ENV','dev')

if ENV == 'dev':
    load_dotenv('.env.dev')
elif ENV == 'test':
    load_dotenv('.env.test')
elif ENV == 'prod':
    load_dotenv('.env.prod')
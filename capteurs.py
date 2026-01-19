import random

def lire_capteurs():
    return {
        "frequence_cardiaque": random.randint(60,120),
        "spo2": random.randint(88,100),
        "temperature_ambiante": random.randint(25,35),
        "humidite": random.randint(60,90)
    }
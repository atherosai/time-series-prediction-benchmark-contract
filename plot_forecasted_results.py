import pandas as pd
from base_config import config
import matplotlib.pyplot as plt

def plot_forecasted_results(
        training_data,
        testing_data,
        forecast,
        with_confidence_intervals=False
    ):

    plt.title(f" {config['ticker']} - horní a dolní interval spolehlivosti predikce")

    plot_args = (
        training_data['date'], training_data['open'], 'blue',
        testing_data['date'], testing_data['open'], 'red',
        forecast['ds'], forecast['yhat'], 'violet'
    )

    if (with_confidence_intervals):
        plot_args = plot_args + (
            forecast['ds'], forecast['yhat_lower'], 'cyan',
            forecast['ds'], forecast['yhat_upper'], 'cyan'
        )

    plt.plot(
        *plot_args
    )

    plt.legend(
        ['Trénovací datový soubor', 'Testovací datový soubor', 'Predikce', 'Horní hranice predikce', 'Dolní hranice predikce']
    )

    plt.xlabel(f" Datum")
    plt.ylabel(f" {config['ticker']} cena při otevření burzy $")

    plt.show()
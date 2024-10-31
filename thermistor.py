import machine
import time
import math

# Configuration
thermistor_pin = 26  # Analog input pin for the thermistor

# Set up ADC on the specified pin
adc = machine.ADC()
thermistor = adc.channel(pin=thermistor_pin)

# Thermistor parameters
reference_resistance = 10000  # Resistance of the fixed resistor (10k)
nominal_temperature = 25  # Nominal temperature for the thermistor (in Celsius)
beta_value = 3950  # Beta value of the thermistor (look for it in the datasheet)

# Function to calculate temperature from thermistor resistance
def calculate_temperature(resistance):
    steinhart = resistance / reference_resistance
    steinhart = math.log(steinhart)
    steinhart /= beta_value
    steinhart += 1.0 / (nominal_temperature + 273.15)
    steinhart = 1.0 / steinhart
    steinhart -= 273.15  # Convert to Celsius
    return steinhart

# Infinite loop to measure and display temperature
while True:
    # Read ADC value
    adc_value = thermistor()

    # Convert ADC value to resistance using voltage divider formula
    thermistor_resistance = reference_resistance * (1 / ((4095 / adc_value) - 1))

    # Calculate temperature
    temperature = calculate_temperature(thermistor_resistance)

    # Display the result
    print("Temperature: {:.2f} °C".format(temperature))

    # Optional: Add a delay to control the measurement rate
    time.sleep(1)

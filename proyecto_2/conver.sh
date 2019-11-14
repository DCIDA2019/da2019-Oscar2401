#! /bin/bash

echo "Introduce el archivo MIDI a convertir a MP3"

read arcchivo

timidity -Ow archivo 

lame archivo 


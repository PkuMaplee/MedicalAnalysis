echo "Downloading......"

wget --no-check-certificate https://github.com/monaen/MedicalAnalysis/raw/data/Lat0001-1000_processed.zip

echo "Unzipping......"

unzip x Lat0001-1000_processed.zip && rm -f Lat0001-1000_processed.zip

echo "Done."

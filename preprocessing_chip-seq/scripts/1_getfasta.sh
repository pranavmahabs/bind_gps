#!/bin/bash

genome_dir=../genomes
peaks_dir=../peaks
peaks_fasta_dir=../peaks_fasta

while IFS=$',' read -r file
do
echo "Processing ${file}..."
name=`echo $file | cut -d '.' -f 1,2`
bedtools getfasta -fi ${genome_dir}/dm6.fa -bed ${peaks_dir}/${file} -fo ${peaks_fasta_dir}/${name}.fasta
done < "1_files_to_getfasta.csv"

echo "Done!"

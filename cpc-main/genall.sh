# Purpose: generate all the data for the paper
# Usage: bash genall.sh
# i have results for nov 30 exp. 
# python calc_emission_inference.py -device cuda -interface pipeline -task_type fill-mask
# python calc_emission_inference.py -device cuda -interface manually-just-inference -task_type fill-mask
# python calc_emission_inference.py -device cuda -interface manually-just-inference-extract -task_type fill-mask
# python calc_emission_inference.py -device cuda -interface manually-just-extract -task_type fill-mask
# python calc_emission_inference.py -device cuda -interface manually -task_type fill-mask
# python calc_emission_inference.py -device cuda -interface manually-just-data -task_type fill-mask

# python calc_emission_inference.py -device cpu -interface pipeline -task_type fill-mask
# python calc_emission_inference.py -device cpu -interface manually-just-inference -task_type fill-mask
# python calc_emission_inference.py -device cpu -interface manually-just-inference-extract -task_type fill-mask
# python calc_emission_inference.py -device cpu -interface manually-just-extract -task_type fill-mask
# python calc_emission_inference.py -device cpu -interface manually -task_type fill-mask
# python calc_emission_inference.py -device cpu -interface manually-just-data -task_type fill-mask

# python calc_emission_inference.py -device cuda -interface pipeline -task_type text-generation
# python calc_emission_inference.py -device cuda -interface manually-just-inference -task_type text-generation
# python calc_emission_inference.py -device cuda -interface manually-just-inference-extract -task_type text-generation
# python calc_emission_inference.py -device cuda -interface manually-just-extract -task_type text-generation
# python calc_emission_inference.py -device cuda -interface manually -task_type text-generation
# python calc_emission_inference.py -device cuda -interface manually-just-data -task_type text-generation

# python calc_emission_inference.py -device cpu -interface pipeline -task_type text-generation
# python calc_emission_inference.py -device cpu -interface manually-just-inference -task_type text-generation
# python calc_emission_inference.py -device cpu -interface manually-just-inference-extract -task_type text-generation
# python calc_emission_inference.py -device cpu -interface manually-just-extract -task_type text-generation
# python calc_emission_inference.py -device cpu -interface manually -task_type text-generation
# python calc_emission_inference.py -device cpu -interface manually-just-data -task_type text-generation


## graphing


python graphing.py -device cuda -interface pipeline -task_type fill-mask
python graphing.py -device cuda -interface manually-just-inference -task_type fill-mask
python graphing.py -device cuda -interface manually-just-inference-extract -task_type fill-mask
python graphing.py -device cuda -interface manually-just-extract -task_type fill-mask
python graphing.py -device cuda -interface manually -task_type fill-mask
python graphing.py -device cuda -interface manually-just-data -task_type fill-mask

python graphing.py -device cpu -interface pipeline -task_type fill-mask
python graphing.py -device cpu -interface manually-just-inference -task_type fill-mask
python graphing.py -device cpu -interface manually-just-inference-extract -task_type fill-mask
python graphing.py -device cpu -interface manually-just-extract -task_type fill-mask
python graphing.py -device cpu -interface manually -task_type fill-mask
python graphing.py -device cpu -interface manually-just-data -task_type fill-mask

python graphing.py -device cuda -interface pipeline -task_type text-generation
python graphing.py -device cuda -interface manually-just-inference -task_type text-generation
python graphing.py -device cuda -interface manually-just-inference-extract -task_type text-generation
python graphing.py -device cuda -interface manually-just-extract -task_type text-generation
python graphing.py -device cuda -interface manually -task_type text-generation
python graphing.py -device cuda -interface manually-just-data -task_type text-generation

python graphing.py -device cpu -interface pipeline -task_type text-generation
python graphing.py -device cpu -interface manually-just-inference -task_type text-generation
python graphing.py -device cpu -interface manually-just-inference-extract -task_type text-generation
python graphing.py -device cpu -interface manually-just-extract -task_type text-generation
python graphing.py -device cpu -interface manually -task_type text-generation
python graphing.py -device cpu -interface manually-just-data -task_type text-generation

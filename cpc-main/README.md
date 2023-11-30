
* Install requirements:

```
pip install -r requirements.txt --upgrade
```

* Use the next command to make sure the func will be able to call the next files:
  
```
chmod 777 ./nvkillprocess.sh
chmod 777 ./nvmodelprofile.sh
```

* Make sure you have an access to file "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj" in order to get cpu power data.
It possoble to change permission level with the next command:

```
sudo chmod -R a+r /sys/class/powercap/intel-rapl
```

* Now we want to check that everything works fine, run the next command and check the result files:
  
```
python calc_emission_inference.py --test
```

* The output include 4 files that should have the same structure (with diff data...) as the example below:

1.nvidia_output_#.log :

```
index, timestamp, power.draw [W], clocks.current.sm [MHz], clocks.current.memory [MHz], clocks.current.graphics [MHz]
0, 2023/08/21 20:01:54.331, 25.19 W, 1807 MHz, 7300 MHz, 1807 MHz
0, 2023/08/21 20:01:55.332, 76.86 W, 1965 MHz, 7300 MHz, 1965 MHz
```

2.pyjoules_output_#.log :
```
begin timestamp : 1692662514.3228672; tag : query; duration : 0.21776676177978516; package_0 : 8146463.0; core_0 : 7768108.0; uncore_0 : 0.0; nvidia_gpu_0 : 8093
begin timestamp : 1692662514.5413787; tag : query; duration : 0.004315376281738281; package_0 : 509947.0; core_0 : 503295.0; uncore_0 : 0.0; nvidia_gpu_0 : 0
begin timestamp : 1692662514.5462317; tag : query; duration : 0.0038220882415771484; package_0 : 374632.0; core_0 : 370055.0; uncore_0 : 0.0; nvidia_gpu_0 : 0
```
3.err_#.log  ( that will include errors and also timestamp that important to seperate the data according model on postprocessing )

4.nvidia_spec_#.log (data about the GPU - kust invida-smi output)

NOW WE CAN USE THE SCRIPT IN ORDER TO MEASURE POWER CONSUMPTION :)

* Use the next command to execute 4 expirements:

```
python calc_emission_inference.py -device "cuda" -interface "pipeline" -task_type "fill-mask" 

python calc_emission_inference.py -device "cuda" -interface "manually-just-data" -task_type "fill-mask" 

python calc_emission_inference.py -device "cuda" -interface "manually-just-inference" -task_type "fill-mask" 

python calc_emission_inference.py -device "cuda" -interface "manually-just-inference-extract" -task_type "fill-mask"
```

---- 
```
sudo apt-get upgrade 
sudo apt-get update 
sudo apt install python3-pip
pip install -r requirements.txt 
```
----- 

```
python calc_emission_inference.py -device cuda -interface pipeline -task_type fill-mask
python calc_emission_inference.py -device cuda -interface manually-just-inference -task_type fill-mask
python calc_emission_inference.py -device cuda -interface manually-just-inference-extract -task_type fill-mask
python calc_emission_inference.py -device cuda -interface manually-just-extract -task_type fill-mask
python calc_emission_inference.py -device cuda -interface manually -task_type fill-mask
python calc_emission_inference.py -device cuda -interface manually-just-data -task_type fill-mask

python calc_emission_inference.py -device cpu -interface pipeline -task_type fill-mask
python calc_emission_inference.py -device cpu -interface manually-just-inference -task_type fill-mask
python calc_emission_inference.py -device cpu -interface manually-just-inference-extract -task_type fill-mask
python calc_emission_inference.py -device cpu -interface manually-just-extract -task_type fill-mask
python calc_emission_inference.py -device cpu -interface manually -task_type fill-mask
python calc_emission_inference.py -device cpu -interface manually-just-data -task_type fill-mask
```

```
python calc_emission_inference.py -device cuda -interface pipeline -task_type text-generation
python calc_emission_inference.py -device cuda -interface manually-just-inference -task_type text-generation
python calc_emission_inference.py -device cuda -interface manually-just-inference-extract -task_type text-generation
python calc_emission_inference.py -device cuda -interface manually-just-extract -task_type text-generation
python calc_emission_inference.py -device cuda -interface manually -task_type text-generation
python calc_emission_inference.py -device cuda -interface manually-just-data -task_type text-generation

python calc_emission_inference.py -device cpu -interface pipeline -task_type text-generation
python calc_emission_inference.py -device cpu -interface manually-just-inference -task_type text-generation
python calc_emission_inference.py -device cpu -interface manually-just-inference-extract -task_type text-generation
python calc_emission_inference.py -device cpu -interface manually-just-extract -task_type text-generation
python calc_emission_inference.py -device cpu -interface manually -task_type text-generation
python calc_emission_inference.py -device cpu -interface manually-just-data -task_type text-generation
```



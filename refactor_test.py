import feffi
import os

feffi.parameters.define_parameters(
	user_config={'final_time':20},
	config_file=os.path.join('config', 'default.yml')
)

if __name__ == '__main__':
	feffi.parameters.parse_commandline_args()

print(feffi.parameters.config)

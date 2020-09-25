import feffi

feffi.parameters.define_parameters(
	user_config={'final_time':20},
	config_file='config.yml'
)
print(feffi.parameters.config)

if __name__ == '__main__':
	feffi.parameters.parse_commandline_args()

print(feffi.parameters.config)

#include "ismrmrd/dataset.h"

// for memcpy and free in older compilers
#include <string.h>
#include <stdlib.h>
#include <stdexcept>
#include "ismrmrd/NHLBICompression.h"

namespace ISMRMRD {
//
// Dataset class implementation
//
// Constructor
Dataset::Dataset(const char* filename, const char* groupname, bool create_file_if_needed)
{
    // TODO error checking and exception throwing
    // Initialize the dataset
    int status;
    status = ismrmrd_init_dataset(&dset_, filename, groupname);
    if (status != ISMRMRD_NOERROR) {
        throw std::runtime_error(build_exception_string());
    }
    // Open the file
    status = ismrmrd_open_dataset(&dset_, create_file_if_needed);
    if (status != ISMRMRD_NOERROR) {
        throw std::runtime_error(build_exception_string());
    }
}

// Destructor
Dataset::~Dataset()
{
    ismrmrd_close_dataset(&dset_);
}

// XML Header
void Dataset::writeHeader(const std::string &xmlstring)
{
    int status = ismrmrd_write_header(&dset_, xmlstring.c_str());
    if (status != ISMRMRD_NOERROR) {
        throw std::runtime_error(build_exception_string());
    }
}

void Dataset::readHeader(std::string& xmlstring){
    char * temp = ismrmrd_read_header(&dset_);
    if (NULL == temp) {
        throw std::runtime_error(build_exception_string());
    } else {
        xmlstring = std::string(temp);
        free(temp);
    }
}

// Acquisitions
void Dataset::appendAcquisition(const Acquisition &acq)
{	
    int status = ismrmrd_append_acquisition(&dset_, reinterpret_cast<const ISMRMRD_Acquisition*>(&acq));
    if (status != ISMRMRD_NOERROR) {
        throw std::runtime_error(build_exception_string());
    }
}

void Dataset::appendAcquisitionCompressed(Acquisition &acq, float local_tolerance)
{
	std::vector<uint8_t> serialized_buffer;
	int segments = acq.getHead().active_channels*4;
	//int segments = 1;			
	int segment_size = acq.getHead().active_channels*acq.getHead().number_of_samples*2/segments;
	if (acq.getHead().active_channels*acq.getHead().number_of_samples*2%segments) {
    	throw std::runtime_error(build_exception_string());
	}
	for(int ch = 0; ch < segments; ch ++){
        std::vector<float> input_data((float*)&acq.getDataPtr()[0]+ch*segment_size, (float*)&acq.getDataPtr()[0]+(ch+1)*segment_size);
        CompressedBuffer<float> comp_buffer(input_data, local_tolerance);
		std::vector<uint8_t> serialized = comp_buffer.serialize();
		if(ch == 0){serialized_buffer = serialized;}
		else{serialized_buffer.insert(serialized_buffer.end(), serialized.begin(), serialized.end());}
	}
	size_t buffer_size = serialized_buffer.size()+sizeof(size_t)+1;
	memcpy(acq.getDataPtr(), &buffer_size, sizeof(size_t)); //bufer size is in bytes
	memcpy((size_t*)&acq.getDataPtr()[0]+1, &serialized_buffer[0], serialized_buffer.size());
	acq.setFlag(ISMRMRD_ACQ_COMPRESSION2);

    int status = ismrmrd_append_acquisition(&dset_, reinterpret_cast<const ISMRMRD_Acquisition*>(&acq));
    if (status != ISMRMRD_NOERROR) {
        throw std::runtime_error(build_exception_string());
    }
}

void Dataset::readAcquisition(uint32_t index, Acquisition & acq) {
    int status = ismrmrd_read_acquisition(&dset_, index, reinterpret_cast<ISMRMRD_Acquisition*>(&acq));
    if (status != ISMRMRD_NOERROR) {
        throw std::runtime_error(build_exception_string());
    }
	if(acq.isFlagSet(ISMRMRD_ACQ_COMPRESSION2)){
		size_t buffer_size = ismrmrd_size_of_acquisition_data(reinterpret_cast<const ISMRMRD_Acquisition*>(&acq));

		std::vector<uint8_t> serialized(buffer_size,0);
		memcpy(&serialized[0],&acq.getDataPtr()[0],buffer_size);
		acq.clearFlag(ISMRMRD_ACQ_COMPRESSION2);
		ismrmrd_make_consistent_acquisition(reinterpret_cast<ISMRMRD_Acquisition*>(&acq));
		serialized.erase(serialized.begin(),serialized.begin()+sizeof(size_t));

		float *d_ptr = (float*) acq.getDataPtr();
		size_t bytes_needed = 0;
		size_t total_size = 0;
		CompressedBuffer<float> comp;

		bytes_needed = comp.deserialize(serialized);
		serialized.erase(serialized.begin(),serialized.begin()+bytes_needed);
        for (size_t i = 0; i < comp.size(); i++) {
            d_ptr[i] = comp[i]; //This uncompresses sample by sample into the uncompressed array
        }
		total_size += comp.size();
		while(total_size < size_t(2*acq.getHead().number_of_samples*acq.getHead().active_channels)){
            CompressedBuffer<float> comp;
            bytes_needed = comp.deserialize(serialized);
			serialized.erase(serialized.begin(),serialized.begin()+bytes_needed);
            for (size_t i = 0; i < comp.size(); i++) {
                d_ptr[i+total_size] = comp[i]; //This uncompresses sample by sample into the uncompressed array
            }
			total_size += comp.size();
		}
	}
}


uint32_t Dataset::getNumberOfAcquisitions()
{
    uint32_t num = ismrmrd_get_number_of_acquisitions(&dset_);
    return num;
}

// Images
template <typename T>void Dataset::appendImage(const std::string &var, const Image<T> &im)
{
    int status = ismrmrd_append_image(&dset_, var.c_str(), &im.im);
    if (status != ISMRMRD_NOERROR) {
        throw std::runtime_error(build_exception_string());
    }
}

void Dataset::appendImage(const std::string &var, const ISMRMRD_Image *im)
{
    int status = ismrmrd_append_image(&dset_, var.c_str(), im);
    if (status != ISMRMRD_NOERROR) {
        throw std::runtime_error(build_exception_string());
    }
}


void Dataset::appendWaveform(const Waveform &wav) {
    int status = ismrmrd_append_waveform(&dset_,&wav);
    if (status != ISMRMRD_NOERROR){
        throw std::runtime_error(build_exception_string());
    }
}

void Dataset::readWaveform(uint32_t index, Waveform &wav) {
    int status = ismrmrd_read_waveform(&dset_,index,&wav);
    if (status != ISMRMRD_NOERROR){
        throw std::runtime_error(build_exception_string());
    }
}

uint32_t Dataset::getNumberOfWaveforms() {
    return ismrmrd_get_number_of_waveforms(&dset_);
}
// Specific instantiations
template EXPORTISMRMRD void Dataset::appendImage(const std::string &var, const Image<uint16_t> &im);
template EXPORTISMRMRD void Dataset::appendImage(const std::string &var, const Image<int16_t> &im);
template EXPORTISMRMRD void Dataset::appendImage(const std::string &var, const Image<uint32_t> &im);
template EXPORTISMRMRD void Dataset::appendImage(const std::string &var, const Image<int32_t> &im);
template EXPORTISMRMRD void Dataset::appendImage(const std::string &var, const Image<float> &im);
template EXPORTISMRMRD void Dataset::appendImage(const std::string &var, const Image<double> &im);
template EXPORTISMRMRD void Dataset::appendImage(const std::string &var, const Image<complex_float_t> &im);
template EXPORTISMRMRD void Dataset::appendImage(const std::string &var, const Image<complex_double_t> &im);


template <typename T> void Dataset::readImage(const std::string &var, uint32_t index, Image<T> &im) {
    int status = ismrmrd_read_image(&dset_, var.c_str(), index, &im.im);
    if (status != ISMRMRD_NOERROR) {
        throw std::runtime_error(build_exception_string());
    }
}

// Specific instantiations
template EXPORTISMRMRD void Dataset::readImage(const std::string &var, uint32_t index, Image<uint16_t> &im);
template EXPORTISMRMRD void Dataset::readImage(const std::string &var, uint32_t index, Image<int16_t> &im);
template EXPORTISMRMRD void Dataset::readImage(const std::string &var, uint32_t index, Image<uint32_t> &im);
template EXPORTISMRMRD void Dataset::readImage(const std::string &var, uint32_t index, Image<int32_t> &im);
template EXPORTISMRMRD void Dataset::readImage(const std::string &var, uint32_t index, Image<float> &im);
template EXPORTISMRMRD void Dataset::readImage(const std::string &var, uint32_t index, Image<double> &im);
template EXPORTISMRMRD void Dataset::readImage(const std::string &var, uint32_t index, Image<complex_float_t> &im);
template EXPORTISMRMRD void Dataset::readImage(const std::string &var, uint32_t index, Image<complex_double_t> &im);

uint32_t Dataset::getNumberOfImages(const std::string &var)
{
    uint32_t num =  ismrmrd_get_number_of_images(&dset_, var.c_str());
    return num;
}


// NDArrays
template <typename T> void Dataset::appendNDArray(const std::string &var, const NDArray<T> &arr)
{
    int status = ismrmrd_append_array(&dset_, var.c_str(), &arr.arr);
    if (status != ISMRMRD_NOERROR) {
        throw std::runtime_error(build_exception_string());
    }
}

// Specific instantiations
template EXPORTISMRMRD void Dataset::appendNDArray(const std::string &var, const NDArray<uint16_t> &arr);
template EXPORTISMRMRD void Dataset::appendNDArray(const std::string &var, const NDArray<int16_t> &arr);
template EXPORTISMRMRD void Dataset::appendNDArray(const std::string &var, const NDArray<uint32_t> &arr);
template EXPORTISMRMRD void Dataset::appendNDArray(const std::string &var, const NDArray<int32_t> &arr);
template EXPORTISMRMRD void Dataset::appendNDArray(const std::string &var, const NDArray<float> &arr);
template EXPORTISMRMRD void Dataset::appendNDArray(const std::string &var, const NDArray<double> &arr);
template EXPORTISMRMRD void Dataset::appendNDArray(const std::string &var, const NDArray<complex_float_t> &arr);
template EXPORTISMRMRD void Dataset::appendNDArray(const std::string &var, const NDArray<complex_double_t> &arr);

void Dataset::appendNDArray(const std::string &var, const ISMRMRD_NDArray *arr)
{
    int status = ismrmrd_append_array(&dset_, var.c_str(), arr);
    if (status != ISMRMRD_NOERROR) {
        throw std::runtime_error(build_exception_string());
    }
}

template <typename T> void Dataset::readNDArray(const std::string &var, uint32_t index, NDArray<T> &arr) {
    int status = ismrmrd_read_array(&dset_, var.c_str(), index, &arr.arr);
    if (status != ISMRMRD_NOERROR) {
        throw std::runtime_error(build_exception_string());
    }
}

// Specific instantiations
template EXPORTISMRMRD void Dataset::readNDArray(const std::string &var, uint32_t index, NDArray<uint16_t> &arr);
template EXPORTISMRMRD void Dataset::readNDArray(const std::string &var, uint32_t index, NDArray<int16_t> &arr);
template EXPORTISMRMRD void Dataset::readNDArray(const std::string &var, uint32_t index, NDArray<uint32_t> &arr);
template EXPORTISMRMRD void Dataset::readNDArray(const std::string &var, uint32_t index, NDArray<int32_t> &arr);
template EXPORTISMRMRD void Dataset::readNDArray(const std::string &var, uint32_t index, NDArray<float> &arr);
template EXPORTISMRMRD void Dataset::readNDArray(const std::string &var, uint32_t index, NDArray<double> &arr);
template EXPORTISMRMRD void Dataset::readNDArray(const std::string &var, uint32_t index, NDArray<complex_float_t> &arr);
template EXPORTISMRMRD void Dataset::readNDArray(const std::string &var, uint32_t index, NDArray<complex_double_t> &arr);

uint32_t Dataset::getNumberOfNDArrays(const std::string &var)
{
    uint32_t num = ismrmrd_get_number_of_arrays(&dset_, var.c_str());
    return num;
}

} // namespace ISMRMRD

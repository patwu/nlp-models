#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <cstring>

namespace tensorflow {

REGISTER_OP("Toids")
    .Input("sentence: string")
    .Output("charids: int32")
    .Output("length : int32")
    .Doc("Convert sentence characters to charids, truncate the length to 256");

class ToIdsOp : public OpKernel {
    public:
        explicit ToIdsOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {

        const Tensor& input_tensor = context->input(0);
        auto input = input_tensor.flat<string>();

        Tensor* ids_tensor = NULL;
        Tensor* len_tensor = NULL;
        //<BEGIN> and <END> to password ids, <BEGIN>=95, <END>=96
        TensorShape shape({TRUNCATE_LEN});//TensorShape shape({TRUNCATE_LEN+2});
        TensorShape shape1({1});

        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &ids_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, shape1, &len_tensor));

        auto ids = ids_tensor->tensor<int,1>();
        auto len = len_tensor->tensor<int,1>();
       
        int i,j;
        string sentence = input(0);
        uint32_t size=sentence.length();
        if(size != TRUNCATE_LEN){
            size=TRUNCATE_LEN;
        }
        for(i=0;i<size+1;i++){ 
            if (sentence.length()>i){
                for (j=0;j<strlen(printable_char);j++){
                    if(sentence[i]==printable_char[j]){
                        ids(i)=j; //ids(i+1)=j;
                        break;
                    }
                }
                if(j==strlen(printable_char)){
                    ids(i)=46;//ids(i+1)=46; //*
                }
            }else{
                ids(i)=0; //ids(i+1)=0;
            }
        }
        //ids(0)=69 + 1;//<BEGIN>
        //ids(size+1)=70;//<END>
        //len(0)=size+2;
        len(0)=size;

    }

    private:
    uint32_t TRUNCATE_LEN=1014; //<BEGIN> sentence <EOS> 254
    const char *printable_char=" 0123456789abcdefghijklmnopqrstuvwxyz`~!@#$%^&*()-_=+[{]}\\|;:\",<.>/?'";

};

REGISTER_KERNEL_BUILDER(Name("Toids").Device(DEVICE_CPU), ToIdsOp);

}  // namespace tensorflow

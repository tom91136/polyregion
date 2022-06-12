FROM centos:7
COPY centos7-setup-llvm.sh /centos7-setup-llvm.sh
RUN chmod +x /centos7-setup-llvm.sh
RUN cd; /centos7-setup-llvm.sh setup-build
RUN cd; /centos7-setup-llvm.sh setup-llvm
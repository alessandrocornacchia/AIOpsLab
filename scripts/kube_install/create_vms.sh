#!/bin/bash
# This is a script to create VMs for the k8s cluster. DRAFT, check before running

# Create bridge network for the cluster

USER=albaraa
NET=kashef-${USER}
IP_ADDR=10.180.100.4
VM=kashef-${USER}-w1

# create VMs associated with that network
# check first the IP address is in the network range
incus launch k8s $VM --vm \
-c limits.cpu=32 \
-c limits.memory=16GiB \
-d root,size=100GiB \
--network $NET \
-d eth0,ipv4.address=$IP_ADDR

incus list

# create and attach storage

# NOTE assume storage kashef-shared is created

# attach volume if you want from host machine
# sudo incus storage volume attach default kashef-shared kashef-k8s-w1 /shared-data



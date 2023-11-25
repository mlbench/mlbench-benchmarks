#!/bin/sh

PERMIT_ROOT_LOGIN=yes
MY_NAME=root

ssh-keygen -f /.sshd/host_keys/host_rsa_key -C '' -N '' -t rsa
ssh-keygen -f /.sshd/host_keys/host_dsa_key -C '' -N '' -t dsa

create_ssh_key() {
  user=$1
  mkdir -p /.sshd/user_keys/$user
  chmod 700 /.sshd/user_keys/$user
  chown $user:$user /.sshd/user_keys/$user
  if ! [ -z "$(ls -A /ssh-key/root)" ]; then
      cp /ssh-key/root/* /.sshd/user_keys/$user/
      chmod 600 /.sshd/user_keys/$user/*
      chown $user:$user /.sshd/user_keys/$user/*
  fi
}

create_ssh_key $MY_NAME

# generating sshd_config
cat << EOT > /.sshd/user_keys/$MY_NAME/sshd_config
# Package generated configuration file
# See the sshd_config(5) manpage for details
# What ports, IPs and protocols we listen for
Port 22
# Use these options to restrict which interfaces/protocols sshd will bind to
#ListenAddress ::
#ListenAddress 0.0.0.0
Protocol 2
PidFile /.sshd/user_keys/$MY_NAME/sshd.pid
# HostKeys for protocol version 2
HostKey /.sshd/host_keys/host_rsa_key
HostKey /.sshd/host_keys/host_dsa_key
#Privilege Separation is turned on for security
UsePrivilegeSeparation no
# Lifetime and size of ephemeral version 1 server key
KeyRegenerationInterval 3600
ServerKeyBits 768
# Logging
SyslogFacility AUTH
LogLevel INFO
# Authentication:
LoginGraceTime 120
PermitRootLogin $PERMIT_ROOT_LOGIN
StrictModes yes
RSAAuthentication yes
PubkeyAuthentication yes
AuthorizedKeysFile /.sshd/user_keys/%u/authorized_keys
# Don't read the user's ~/.rhosts and ~/.shosts files
IgnoreRhosts yes
# For this to work you will also need host keys in /etc/ssh_known_hosts
RhostsRSAAuthentication no
# similar for protocol version 2
HostbasedAuthentication no
# Uncomment if you don't trust ~/.ssh/known_hosts for RhostsRSAAuthentication
#IgnoreUserKnownHosts yes
# To enable empty passwords, change to yes (NOT RECOMMENDED)
PermitEmptyPasswords no
# Change to yes to enable challenge-response passwords (beware issues with
# some PAM modules and threads)
ChallengeResponseAuthentication no
X11Forwarding yes
X11DisplayOffset 10
PrintMotd no
PrintLastLog yes
TCPKeepAlive yes
#UseLogin no
# Allow client to pass locale environment variables
AcceptEnv LANG LC_*
Subsystem sftp /usr/lib/openssh/sftp-server
# Set this to 'yes' to enable PAM authentication, account processing,
# and session processing. If this is enabled, PAM authentication will
# be allowed through the ChallengeResponseAuthentication and
# PasswordAuthentication.  Depending on your PAM configuration,
# PAM authentication via ChallengeResponseAuthentication may bypass
# the setting of "PermitRootLogin without-password".
# If you just want the PAM account and session checks to run without
# PAM authentication, then enable this but set PasswordAuthentication
# and ChallengeResponseAuthentication to 'no'.
UsePAM no
# we need this to set various variables (LD_LIBRARY_PATH etc.) for users
# since sshd wipes all previously set environment variables when opening
# a new session
PermitUserEnvironment yes
EOT

#cat << EOT > /$MY_NAME/.ssh/config
cat << EOT > /etc/ssh/ssh_config
StrictHostKeyChecking no
IdentityFile /.sshd/user_keys/$MY_NAME/id_rsa
Port 22
UserKnownHostsFile=/dev/null
EOT

#prepare run dir
if [ ! -d "/var/run/sshd" ]; then
  mkdir -p /var/run/sshd
fi
# EOT
exec "$@"
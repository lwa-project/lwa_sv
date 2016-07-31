
/usr/sbin/ntpdate 10.1.1.50 ntp.ubuntu.com pool.ntp.org
apt-get install -y ntp
cp ntp.conf /etc/ntp.conf
restart ntp
ntpq -p

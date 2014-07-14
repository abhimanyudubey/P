/** file:   tinyLockClient.c
 ** brief:  tinyLockClient - MEX implementation
 ** author: Andrea Vedaldi
 **/

#include "mex.h"

#if defined(__WIN32__) || defined(__WIN64__) || defined(__WINDOWS__) || \
    defined(_WIN32) || defined(_WIN64)
#include <io.h>
#include <stdlib.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/select.h>
#include <sys/time.h>
#include <unistd.h>
#include <netdb.h>
#endif

#include <sys/types.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#define close _close
#endif

/* Return value:
 -1: error occured
  0: timed out
 >0: data ready to be read
*/

int
recvfromTimeOut(int socket, long sec, long usec)
{
  struct timeval timeout ;
  fd_set fds ;

  timeout.tv_sec = sec ;
  timeout.tv_usec = usec ;

  FD_ZERO (&fds) ;
  FD_SET (socket, &fds) ;

  return select(socket+1, &fds, 0, 0, &timeout);
}

void
mexFunction(int nout, mxArray *out[],
            int nin, const mxArray *in[])
{
  char address [1024] ;
  char port [1024] ;
  char key [1024] ;

  struct addrinfo servHints ;
  struct addrinfo *servInfo ;

  int status ;

  enum {IN_HOST, IN_PORT, IN_KEY} ;
  enum {OUT_GO} ;

  if (nin != 3) {
    mexErrMsgTxt("Three arguments required") ;
  }

  if (mxGetString(in[IN_HOST], address, sizeof(address))) {
    mexErrMsgTxt("HOST must be a string") ;
  }
  if (mxGetString(in[IN_PORT], port, sizeof(port))) {
    mexErrMsgTxt("PORT must be a string") ;
  }
  if (mxGetString(in[IN_KEY], key, sizeof(key))) {
    mexErrMsgTxt("KEY must be a string") ;
  }

  /* IN address of server */
  memset(&servHints, 0, sizeof(servHints));
  servHints.ai_family   = AF_INET ; /* AF_UNSPEC */
  servHints.ai_socktype = SOCK_DGRAM ;

  status = getaddrinfo(address, port, &servHints, &servInfo) ;

  if (status) {
    mexPrintf(gai_strerror(status)) ;
    mexErrMsgTxt("getaddrinfo") ;
  }

#if 1
  {
    struct addrinfo *p ;
    for(p = servInfo ; p != NULL ; p = p->ai_next) {
      void *addr;
      char *ipver;
      char ipstr[INET6_ADDRSTRLEN];
      int port ;

      /* get the pointer to the address itself,
       * different fields in IPv4 and IPv6: */
      if (p->ai_family == AF_INET) { /* IPv4 */
        struct sockaddr_in *ipv4 = (struct sockaddr_in *) p->ai_addr ;
        addr = &(ipv4->sin_addr);
        port = ntohs(ipv4->sin_port) ;
        ipver = "IPv4";
      } else { /* IPv6 */
        struct sockaddr_in6 *ipv6 = (struct sockaddr_in6 *)p->ai_addr;
        addr = &(ipv6->sin6_addr);
        port = ntohs(ipv6->sin6_port) ;
        ipver = "IPv6";
      }

      /* convert the IP to a string and print it: */
      inet_ntop(p->ai_family, addr, ipstr, sizeof ipstr);
      mexPrintf("  %s: %s (port %d)\n", ipver, ipstr, port);
    }
  }
#endif

  {
    char answer [1024] ;
    int unsigned answerLength = sizeof(answer) ;
    int answerRecvLength ;
    int clientSocket ;
    socklen_t fromLength = sizeof(struct sockaddr_in) ;
    int numAttemptsLeft = 5 ;

    memset(answer, 0, answerLength) ;

    clientSocket = socket(servInfo->ai_family,
                          servInfo->ai_socktype,
                          servInfo->ai_protocol) ;

    if (clientSocket < 0) {
      mexErrMsgTxt("socket()") ;
    }

    while (numAttemptsLeft -- > 0) {

      status = sendto(clientSocket,
                      key,
                      strlen(key),
                      0,
                      servInfo->ai_addr,
                      sizeof(struct sockaddr_in)) ;

      status = recvfromTimeOut(clientSocket, 5, 0) ;

      if (status < 0) break ;
      if (status == 0) {
        mexPrintf("Resending request due to timeout (%d left)\n", 
                  numAttemptsLeft) ;
        continue ;
      }

      answerRecvLength = recvfrom(clientSocket,
                                  answer,
                                  answerLength,
                                  0,
                                  servInfo->ai_addr,
                                  &fromLength) ;
      break ;
      /* mexPrintf("answ %d: '%s'\n", answerRecvLength, answer) ; */
    }

    close(clientSocket) ;
    out[OUT_GO] = mxCreateLogicalScalar(strcmp(answer, "stop") != 0) ;
  }

  freeaddrinfo(servInfo) ;
}

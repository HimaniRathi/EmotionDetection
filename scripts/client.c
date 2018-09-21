#include <stdio.h>
#include <stdlib.h>

#include <netdb.h>
#include <netinet/in.h>

#include <string.h>

int main(int argc, char *argv[]) {
    

    if (argc < 3) {
        fprintf(stderr,"usage %s hostname(s) port [command]\n", argv[0]);
        exit(0);
    }

    int portno = atoi(argv[2]);

    /* Create a socket point */
    

    
    char *token;
    printf("Hosts: %s\n", argv[1]);
    token = strtok(argv[1], ":");
    
    while( token != NULL ) {
        printf( "host : %s\n", token );
        
        int sockfd, n;
        struct sockaddr_in serv_addr;
        struct hostent *server;

        char buffer[256];
        
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            perror("ERROR opening socket");
            token = strtok(NULL, ":");
            continue;
        }
        server = gethostbyname(token);

        if (server == NULL) {
            fprintf(stderr,"ERROR, no such host:%s\n",server);
            token = strtok(NULL, ":");
            continue;
        }

        bzero((char *) &serv_addr, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr, server->h_length);
        serv_addr.sin_port = htons(portno);

        /* Now connect to the server */
        if (connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
            perror("ERROR connecting to host\n");
            token = strtok(NULL, ":");
            continue;
        }

        /* Now ask for a message from the user, this message
          * will be read by server
        */
        if(argc!=4){
            printf("Please enter the commmand: ");
            bzero(buffer,256);
            fgets(buffer,255,stdin);
        }
        else
        {
            strcpy(buffer,argv[3]);
        }
        /* Send message to the server */
        n = write(sockfd, buffer, strlen(buffer));

        token = strtok(NULL, ":");
    }
    return 0;
}

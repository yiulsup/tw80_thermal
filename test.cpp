#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/select.h>
#include <string.h>

using namespace cv::dnn;	

        
#define BAUDRATE B115200 
#define MODEMDEVICE "/dev/ttyACM0"
#define _POSIX_SOURCE 1 /* POSIX compliant source */
#define FALSE 0
#define TRUE 1

FILE *file;
int fileLen;
char *tmpbuffer;
void openport(void);
void readport(void);
void sendport(void);
int fd = 0;
struct termios oldtp, newtp;
char buffer[512];

unsigned char sending_1[] = {0x02, 0x00, 0x04, 0x00, 0x01, 0x55, 0xaa, 0x03, 0xFA};
unsigned char sending_2[] = {0x02, 0x00, 0x04, 0x01, 0x01, 0x00, 0x5, 0x03, 0x01};
//sending_2 = [0x02, 0x00, 0x04, 0x01, 0x01, 0x00, 0x0a, 0x03, 0x0E]
unsigned char sending_3[] = {0x02, 0x00, 0x04, 0x02, 0x01, 0x00, 0x01, 0x03, 0x06};
unsigned char sending_4[] = {0x02, 0x00, 0x04, 0x02, 0x01, 0x01, 0x01, 0x03, 0x07};

void sendport(void)
{
    int n = 0;
    unsigned char buff;
    int cnt = 0;
    printf("first write \n");
    n = write(fd, sending_2, sizeof(sending_1));
    //usleep(1000);
    printf("second write \n");
    n = write(fd, sending_4, sizeof(sending_1));
   // usleep(1000);
    printf("third write \n");
    n = write(fd, sending_1, sizeof(sending_1));
    while (1) {
        n = read(fd, &buff, 1);
        cnt = cnt + 1;
        if (cnt >= 9) {
            cnt = 0;
            break;
        }
    }
    //usleep(1000);
    printf("fourth write \n");
    n = write(fd, sending_4, sizeof(sending_1));
    //usleep(1000);
    if (n < 0)
    {
        fputs("write() of bytes failed!\n", stderr);
    }
    else
    {
        printf("Image sent successfully %d\n",n);
    }
    
    printf("write success\n");
}

int cnt = 0;
int cnt1 = 0;
int image_cnt = 0;
int begin = 0;
int first = 1;
int dec = 0;
int dec_10 = 0;
unsigned char frame[9600];
int getFlag = 0;
unsigned char rawDataDecimal = 0;

unsigned char buff;

void readport(void) {
    int n = 0;
    
    while (1) { 
        //printf("recv start\n");
        cnt1 = cnt1 + 1;
        
        n = read(fd, &buff, 1);
        if (n == -1) {
            printf("Error in input\n");
            continue;
        }

        if (cnt1 == 1) {
            printf("\nStart frame\n");
        }

        if (cnt1 == 9638) {
            cnt1 = 0;
            printf("\nframe end\n");
        }

        printf("%d : %d \n", cnt1, buff);
        continue;
    }

}

void  process(void)
{   
    int n = 0;
    
    while (1) { 
        //printf("recv start\n");
        n = read(fd, &buff, 1);
        if (n == -1) {
            printf("Error in input\n");
            continue;
        }

        printf("%d ", buff);
        continue;
        cnt1 = cnt1 + 1;

        if (begin == 0 and cnt1 == 1) {
            rawDataDecimal = buff;
            if (rawDataDecimal == 2)
                begin = 1;
            else {
                printf("Error\n");
                begin = 0;
                cnt1 = 0;
                continue;
            }
        }

        int j = 0;
        if (begin == 1 and cnt1 == 20) {
            for (int i=0, j = 0; i<9600; i++, j=j+2) {
                n = read(fd, &buff, 1);
                cnt1 = cnt1 + 1;
                rawDataDecimal = buff;
                printf("%d", buff);
                if (first == 1) {
                    dec_10 = rawDataDecimal * 256;
                    first = 2;
                } 
                else if (first == 2) {
                    first = 1;
                    dec = rawDataDecimal;
                    frame[j] = dec + dec_10;
                    image_cnt = image_cnt + 1;
                }

                if (image_cnt >= 4800) {
                    image_cnt = 0;
                    getFlag = 1;
                    for (int i=0; i<4800; i++)
                        printf("%x", frame[i]);
                }           
            }
        }

        if (cnt1 == 2 and begin == 1) {
            if (rawDataDecimal == 0x25) {
                begin = 1;
            }
            else {
                begin = 0;
                cnt1 = 0;
                continue;
            }

        }

        if (cnt1 == 3 and begin == 1) {
            rawDataDecimal = buff;
            if (rawDataDecimal == 0xA1) {
                begin = 1;

            }
            else {
                begin = 0;
                cnt1 = 0;
                continue;
            }
        }

        if (cnt1 == 9638 and begin == 1) {
            begin = 0;
            cnt1 = 0;
            printf("recived end");
        }
        else 
            continue;
    }
}

void openport(void)
{
    fd = open(MODEMDEVICE, O_RDWR | O_NOCTTY |O_NDELAY );
	printf("Oviya %d\n",fd);
    if (fd <0)
    {
        perror(MODEMDEVICE);         
    }

    printf("file Open Success\n");
                                                                            
    fcntl(fd,F_SETFL,0);
    tcgetattr(fd,&oldtp); /* save current serial port settings */
    bzero(&newtp, sizeof(newtp));
                                                                        
    newtp.c_cflag = BAUDRATE | CRTSCTS | CS8 | CLOCAL | CREAD;
                                                                        
    newtp.c_iflag = IGNPAR | ICRNL;
                                                                        
    newtp.c_oflag = 0;
                                                                        
    newtp.c_lflag = ICANON;
                                                                        
    newtp.c_cc[VINTR]    = 0;     /* Ctrl-c */
    newtp.c_cc[VQUIT]    = 0;     /* Ctrl-\ */
    newtp.c_cc[VERASE]   = 0;     /* del */
    newtp.c_cc[VKILL]    = 0;     /* @ */
    //newtp.c_cc[VEOF]     = 4;     /* Ctrl-d */
    newtp.c_cc[VEOF]     = 0;     /* Ctrl-d */
    newtp.c_cc[VTIME]    = 0;     /* inter-character timer unused */
    newtp.c_cc[VMIN]     = 1;     /* blocking read until 1 character arrives */
    newtp.c_cc[VSWTC]    = 0;     /* '\0' */
    newtp.c_cc[VSTART]   = 0;     /* Ctrl-q */
    newtp.c_cc[VSTOP]    = 0;     /* Ctrl-s */
    newtp.c_cc[VSUSP]    = 0;     /* Ctrl-z */
    newtp.c_cc[VEOL]     = 0;     /* '\0' */
    newtp.c_cc[VREPRINT] = 0;     /* Ctrl-r */
    newtp.c_cc[VDISCARD] = 0;     /* Ctrl-u */
    newtp.c_cc[VWERASE]  = 0;     /* Ctrl-w */
    newtp.c_cc[VLNEXT]   = 0;     /* Ctrl-v */
    newtp.c_cc[VEOL2]    = 0;     /* '\0' */

    printf("Baudrate Setting 115200\n");
                                                                        
}


struct detectionResult
{
	cv::Rect plateRect;
	double confidence;
	int type;
};

void NMS(std::vector<detectionResult>& vResultRect);

int main() {

	cv::Mat img = cv::imread("./prisonw123.png");
    cv::imshow("d", img);
    cv::waitKey(1);

    int uartd = 0;
    pthread_t rx_th;
    int rx_th_id;

    openport();
    sendport();
    readport();
    
	const float confidenceThreshold = 0.24f;
	Net m_net;
	std::string yolo_cfg = "./yolov2-tiny.cfg";
	std::string yolo_weights = "./yolov2-tiny_255000.weights";
	m_net = readNetFromDarknet(yolo_cfg, yolo_weights);
	m_net.setPreferableBackend(DNN_BACKEND_OPENCV);
	m_net.setPreferableTarget(DNN_TARGET_CPU);

	cv::Mat inputBlob = blobFromImage(img, 1 / 255.F, cv::Size(416, 416), cv::Scalar(), true, false); //Convert Mat to batch of images

	m_net.setInput(inputBlob);

	std::vector<cv::Mat> outs;

	cv::Mat detectionMat = m_net.forward();

	std::vector<detectionResult> vResultRect;

	for (int i = 0; i < detectionMat.rows; i++)
	{
		const int probability_index = 5;
		const int probability_size = detectionMat.cols - probability_index;
		float* prob_array_ptr = &detectionMat.at<float>(i, probability_index);
		size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
		float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);
		if (confidence > confidenceThreshold)
		{
			float x_center = detectionMat.at<float>(i, 0) * (float)img.cols;
			float y_center = detectionMat.at<float>(i, 1) * (float)img.rows;
			float width = detectionMat.at<float>(i, 2) * (float)img.cols;
			float height = detectionMat.at<float>(i, 3) * (float)img.rows;
			cv::Point2i p1(round(x_center - width / 2.f), round(y_center - height / 2.f));
			cv::Point2i p2(round(x_center + width / 2.f), round(y_center + height / 2.f));
			cv::Rect2i object(p1, p2);

			detectionResult tmp;
			tmp.plateRect = object;
			tmp.confidence = confidence;
			tmp.type = objectClass;
			vResultRect.push_back(tmp);
		}
	}

	NMS(vResultRect);

	for (int i = 0; i < vResultRect.size(); i++)
	{
		cv::rectangle(img, vResultRect[i].plateRect, cv::Scalar(0, 0, 255), 2);
		printf("index: %d, confidence: %g\n", vResultRect[i].type, vResultRect[i].confidence);
	}

	cv::imshow("img", img);
	cv::waitKey();

	return 1;
}

void NMS(std::vector<detectionResult>& vResultRect)
{
	for (int i = 0; i < vResultRect.size() - 1; i++)
	{
		for (int j = i + 1; j < vResultRect.size(); j++)
		{
			double IOURate = (double)(vResultRect[i].plateRect & vResultRect[j].plateRect).area() / (vResultRect[i].plateRect | vResultRect[j].plateRect).area();
			if (IOURate >= 0.5)
			{
				if (vResultRect[i].confidence > vResultRect[j].confidence) {
					vResultRect.erase(vResultRect.begin() + j);
					j--;
				}
				else {
					vResultRect.erase(vResultRect.begin() + i);
					i--;
					break;
				}
			}
		}
	}
}